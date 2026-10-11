"""Read the free-text description: fill blank categories and flag contradictions.

Policy:
- A logged category is NEVER overwritten. Text is only used to fill a category that is
  missing, and to flag (category_text_mismatch) where the text disagrees with what was logged.
- When the text gives no clear signal the answer is "Unclassified", not a guess.
- Two methods run side by side so they can be compared fairly:
  1. keyword rules: transparent, no training, used to fill blanks and raise flags;
  2. an optional small ML model (TF-IDF over character n-grams + logistic regression).
- The ML predictions for rows that have a logged category are out-of-fold: each row is scored
  by a model that never saw that row's label, so a model cannot "agree" with a wrong label
  just because it memorized it.
"""

import re
from dataclasses import dataclass, field
from functools import cache

import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from field_ops import config
from field_ops.clean import RAW_COLUMNS, UNKNOWN, CleanResult, Ledger

UNCLASSIFIED = config.UNCLASSIFIED

FINAL_COLUMNS = [
    *RAW_COLUMNS,
    "category_source",
    "text_category",
    "dq_flags",
    "dup_count",
    "merged_report_ids",
]

# ---------------------------------------------------------------------------------------------
# Keyword rules
# ---------------------------------------------------------------------------------------------
# Written as a field engineer would list the vocabulary of each system area. Word-start
# patterns keep "batter" matching battery/batteries. Bare ambiguous words ("drift", "log",
# "frame") are deliberately absent: they appear in several categories.
KEYWORDS: dict[str, list[str]] = {
    "Power": [
        r"\bbatter(?:y|ies)\b", r"\bcharg\w*", r"\b(?:under|over)?voltage\b", r"\bpower\b",
        r"\bpsu\b", r"\bregulator\b", r"\bbrown-?out\b",
    ],
    "Communications": [
        r"\bantenna", r"\bradio\b", r"\bnetwork", r"\blink\b", r"\btelemetry",
        r"\bsignal", r"\bpacket", r"\bcomms?\b", r"\bcommunicat\w*", r"\brssi\b",
        r"\blatency\b", r"\bconnection\b", r"\bconnectivity\b", r"\bwireless",
    ],
    "Calibration": [
        r"\bcalibrat\w*", r"\balignment", r"\btolerance", r"\boffset",
        r"\breference clock", r"\btiming", r"\bsettle\w*", r"\bzero[- ]point",
    ],
    "Sensor": [
        r"\bcamera", r"\bthermal", r"\blens", r"\bzoom", r"\bsensor", r"\boptic\w*",
        r"\bdetector", r"\bimag(?:e|ery)\b", r"\bvideo feed", r"\binfrared", r"\bblurr\w*",
    ],
    "Software": [
        r"\bsoftware", r"\bfirmware", r"\bapp\b", r"\bapplication\b", r"\bui\b",
        r"\berror code", r"\be-\d{3}\b", r"\bcrash\w*", r"\bfreez\w*", r"\bfroze\b",
        r"\binstall\w*", r"\bupdate\b", r"\b(?:file|job) upload", r"\bscheduling",
        r"\blog server", r"\bsettings\b", r"\bbug\b", r"\bglitch",
    ],
    "Mechanical": [
        r"\bbelt\b", r"\bbearing", r"\bhousing", r"\bmotor", r"\bmount", r"\bpanel",
        r"\bservo", r"\bfan\b", r"\bcrack\w*", r"\bvibrat\w*", r"\bworn\b",
        r"\bloose\b", r"\bhinge", r"\babnormal noise", r"\bgrinding",
    ],
}  # fmt: skip
_COMPILED = {cat: [re.compile(p) for p in pats] for cat, pats in KEYWORDS.items()}

# Everything after one of these words describes what happened before / caused the fault, not
# the faulty part ("Calibration unit drifts AFTER a firmware update" is a calibration fault).
# Requiring at least one character before the connector keeps "After transport: fan is cracked"
# (context first) from being cut to nothing.
_CAUSE_SPLIT = re.compile(r"^(.+?)\b(?:after|following|caused by|due to|because of)\b")


def main_clause(text: str) -> str:
    match = _CAUSE_SPLIT.match(text)
    return match.group(1) if match else text


@cache
def rule_classify(description: str) -> tuple[str, str]:
    """Return (category, reason). One clear winner -> category, otherwise Unclassified."""
    main = main_clause(description.lower())
    scores = {cat: sum(bool(p.search(main)) for p in pats) for cat, pats in _COMPILED.items()}
    best = max(scores.values())
    if best == 0:
        return UNCLASSIFIED, "no_keyword"
    winners = [cat for cat, score in scores.items() if score == best]
    if len(winners) > 1:
        return UNCLASSIFIED, "ambiguous"
    return winners[0], "keyword"


# ---------------------------------------------------------------------------------------------
# Optional ML model
# ---------------------------------------------------------------------------------------------
ML_THRESHOLD = 0.5  # below this top probability the model abstains (6 classes: chance ~ 0.17)
ML_MIN_TRAIN = 60
ML_MIN_PER_CLASS = 5
ML_C_GRID = [1, 10, 100]
ML_FOLDS = 5


def _make_model() -> object:
    # Character n-grams inside word boundaries tolerate typos ("antennna" still shares most
    # n-grams with "antenna"), which is where keyword rules are weakest.
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2),
        LogisticRegression(max_iter=2000),
    )


def ml_apply(model, texts: pd.Series) -> pd.Series:
    """Category for new texts from a fitted model, abstaining below the confidence threshold."""
    proba = pd.DataFrame(model.predict_proba(texts), index=texts.index, columns=model.classes_)
    return proba.idxmax(axis=1).where(proba.max(axis=1) >= ML_THRESHOLD, UNCLASSIFIED)


def ml_predict(
    texts: pd.Series, logged: pd.Series, seed: int = config.SEED
) -> tuple[pd.DataFrame | None, dict, object | None]:
    """Train on rows with a logged category; predict every row without leaking its own label.

    Training labels are the *logged* categories (noisy: some are wrong), never the ground truth.
    """
    trainable = logged.isin(config.CATEGORIES)
    y = logged[trainable]
    if len(y) < ML_MIN_TRAIN or y.value_counts().min() < ML_MIN_PER_CLASS:
        return None, {"skipped": f"too few labelled rows ({len(y)}) to train"}, None

    folds = StratifiedKFold(ML_FOLDS, shuffle=True, random_state=seed)
    search = GridSearchCV(
        _make_model(),
        {"logisticregression__C": ML_C_GRID},
        scoring="neg_log_loss",
        cv=folds,
    )
    search.fit(texts[trainable], y)
    best = search.best_estimator_  # refit on all labelled rows
    classes = list(best.classes_)

    proba = pd.DataFrame(0.0, index=texts.index, columns=classes)
    proba.loc[trainable] = cross_val_predict(
        clone(best), texts[trainable], y, cv=folds, method="predict_proba"
    )
    if (~trainable).any():
        proba.loc[~trainable] = best.predict_proba(texts[~trainable])

    confidence = proba.max(axis=1)
    category = proba.idxmax(axis=1).where(confidence >= ML_THRESHOLD, UNCLASSIFIED)
    frame = pd.DataFrame({"ml_category": category, "ml_confidence": confidence.round(3)})
    info = {
        "n_train": int(trainable.sum()),
        "c": best.get_params()["logisticregression__C"],
        "cv_log_loss": round(-float(search.best_score_), 3),
        "threshold": ML_THRESHOLD,
        "classes": classes,
    }
    return frame, info, best


# ---------------------------------------------------------------------------------------------
# Applying the result to the cleaned rows
# ---------------------------------------------------------------------------------------------
@dataclass
class ClassifyResult:
    final: pd.DataFrame  # delivered rows, in the same order as CleanResult.clean
    survivors: pd.DataFrame  # cleaned survivor rows (before classification), same order
    predictions: pd.DataFrame  # rule_* and ml_* columns, indexed like ``survivors``
    ml_info: dict
    ml_model: object | None = None  # fitted on all rows with a logged category
    ledger: Ledger = field(default_factory=Ledger)


def _update_flags(flags: pd.Series, add: dict[str, pd.Series], drop: set[str]) -> pd.Series:
    out = []
    for label, base in flags.items():
        kept = [f for f in base.split(";") if f and f not in drop]
        kept += [name for name, mask in add.items() if mask.loc[label]]
        out.append(";".join(kept))
    return pd.Series(out, index=flags.index)


def classify_reports(
    result: CleanResult, *, use_ml: bool = True, seed: int = config.SEED
) -> ClassifyResult:
    survivors = result.rows.loc[result.survivor_index]
    logged = survivors["fault_category"]  # cleaned logged category; Unknown if it was blank
    descriptions = survivors["description"]

    rules = descriptions.map(rule_classify)
    predictions = pd.DataFrame(
        {
            "rule_category": rules.map(lambda r: r[0]),
            "rule_reason": rules.map(lambda r: r[1]),
        }
    )
    ml_info: dict = {"skipped": "disabled"}
    ml_model = None
    if use_ml:
        ml_frame, ml_info, ml_model = ml_predict(descriptions, logged, seed)
        if ml_frame is not None:
            predictions = predictions.join(ml_frame)

    text_category = predictions["rule_category"]
    has_logged = logged != UNKNOWN
    filled = ~has_logged & (text_category != UNCLASSIFIED)
    unclassified = ~has_logged & (text_category == UNCLASSIFIED)
    mismatch = has_logged & (text_category != UNCLASSIFIED) & (text_category != logged)

    final = result.clean.copy()
    final.index = result.survivor_index
    # Only blanks are touched; every logged category passes through unchanged.
    final["fault_category"] = logged.where(has_logged, text_category.where(filled, UNCLASSIFIED))
    final["category_source"] = "logged"
    final.loc[filled, "category_source"] = "filled_from_text"
    final.loc[unclassified, "category_source"] = "unclassified"
    final["text_category"] = text_category
    final["dq_flags"] = _update_flags(
        final["dq_flags"],
        {
            "category_filled_from_text": filled,
            "category_unclassified": unclassified,
            "category_text_mismatch": mismatch,
        },
        drop={"category_unknown"},
    )

    ledger = Ledger()
    example = f"blank -> '{text_category[filled].value_counts().index[0]}'" if filled.any() else ""
    ledger.add(
        "classify", "blank category filled from description (keyword rules)", filled.sum(), example
    )
    ledger.add(
        "classify",
        "blank category left Unclassified (description vague or ambiguous)",
        unclassified.sum(),
        "blank -> 'Unclassified'",
    )
    pair = (
        f"logged '{logged[mismatch].iloc[0]}' vs text '{text_category[mismatch].iloc[0]}'"
        if mismatch.any()
        else ""
    )
    ledger.add(
        "classify",
        "logged category contradicts description -> flagged for review (not changed)",
        mismatch.sum(),
        pair,
    )

    final = final[FINAL_COLUMNS].reset_index(drop=True)
    return ClassifyResult(
        final=final,
        survivors=survivors,
        predictions=predictions,
        ml_info=ml_info,
        ml_model=ml_model,
        ledger=ledger,
    )
