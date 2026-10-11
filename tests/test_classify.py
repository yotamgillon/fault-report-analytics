import numpy as np
import pandas as pd
import pytest

from field_ops import config, evaluate
from field_ops.classify import (
    ML_MIN_TRAIN,
    UNCLASSIFIED,
    classify_reports,
    main_clause,
    ml_predict,
    rule_classify,
)
from field_ops.clean import UNKNOWN, clean_reports
from field_ops.mess import RAW_COLUMNS


# --- keyword rules ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Battery pack drops voltage under load.", "Power"),
        ("Network link loses signal intermittently on startup.", "Communications"),
        ("Calibration unit drifts out of tolerance.", "Calibration"),
        ("Thermal sensor produces a noisy image.", "Sensor"),
        ("Controller firmware freezes during file upload.", "Software"),
        ("Operator reports control software shows error code E-453.", "Software"),
        ("Drive belt is cracked.", "Mechanical"),
        ("BATTERY PACK DROPS VOLTAGE UNDER LOAD.", "Power"),
    ],
)
def test_rules_classify_clear_text(text, expected):
    assert rule_classify(text) == (expected, "keyword")


def test_text_after_a_cause_connector_is_ignored():
    text = "Calibration unit drifts out of tolerance after a firmware update."
    assert main_clause(text.lower()).strip() == "calibration unit drifts out of tolerance"
    assert rule_classify(text)[0] == "Calibration"
    assert rule_classify("Alignment module loses alignment, possibly caused by a hard impact.")[
        0
    ] == ("Calibration")


def test_context_written_first_is_not_cut_away():
    assert rule_classify("After transport: cooling fan is cracked.")[0] == "Mechanical"


@pytest.mark.parametrize(
    "text",
    [
        "Unit not working properly.",
        "Intermittent fault, see attached log.",
        "Same problem as last week.",
        "Something wrong after maintenance, please check.",
        "",
    ],
)
def test_vague_text_is_unclassified_not_guessed(text):
    assert rule_classify(text) == (UNCLASSIFIED, "no_keyword")


def test_two_categories_in_the_main_clause_is_ambiguous_not_guessed():
    assert rule_classify("Camera crashes on startup") == (UNCLASSIFIED, "ambiguous")


# --- applying to rows: never overwrite, only fill blanks, flag contradictions ----------------
def _row(n: int, *, category="Sensor", description="Camera module produces a blurry image.", **kw):
    base = {
        "report_id": f"FR-{10000 + n}", "date_opened": "2026-03-01", "date_closed": "2026-03-05",
        "site": "Lakeside", "system_type": "Sensor Module", "system_id": f"SEN-0{n}",
        "vendor": "Vendor A", "fault_category": category, "severity": "High",
        "description": description, "reported_by": "Ruth Adler", "status": "Closed",
    }  # fmt: skip
    return {**base, **kw}


def _classify(*rows: dict):
    result = clean_reports(pd.DataFrame(rows, columns=RAW_COLUMNS))
    return classify_reports(result, use_ml=False).final.set_index("report_id")


def test_logged_category_is_never_overwritten_but_flagged():
    out = _classify(_row(1, category="Power"))  # text clearly describes a camera
    row = out.loc["FR-10001"]
    assert row["fault_category"] == "Power"
    assert row["category_source"] == "logged"
    assert row["text_category"] == "Sensor"
    assert "category_text_mismatch" in row["dq_flags"]


def test_agreeing_category_is_not_flagged():
    row = _classify(_row(1)).loc["FR-10001"]
    assert "category_text_mismatch" not in row["dq_flags"]
    assert row["fault_category"] == "Sensor"


def test_blank_category_is_filled_from_clear_text():
    row = _classify(_row(1, category="")).loc["FR-10001"]
    assert row["fault_category"] == "Sensor"
    assert row["category_source"] == "filled_from_text"
    assert "category_filled_from_text" in row["dq_flags"]
    assert "category_unknown" not in row["dq_flags"]


def test_blank_category_with_vague_text_stays_unclassified():
    row = _classify(_row(1, category="", description="Not working again.")).loc["FR-10001"]
    assert row["fault_category"] == UNCLASSIFIED
    assert row["category_source"] == "unclassified"
    assert "category_unclassified" in row["dq_flags"]


def test_vague_text_never_contradicts_a_logged_category():
    row = _classify(_row(1, category="Power", description="Not working again.")).loc["FR-10001"]
    assert row["fault_category"] == "Power"
    assert "category_text_mismatch" not in row["dq_flags"]


def test_two_category_text_judged_by_the_faulty_part():
    text = "Alignment module loses alignment after a firmware update."
    agree = _classify(_row(1, category="Calibration", description=text)).loc["FR-10001"]
    assert "category_text_mismatch" not in agree["dq_flags"]


def test_other_flags_survive_classification():
    row = _classify(_row(1, category="Poewr", date_closed="")).loc["FR-10001"]
    assert "closed_no_close_date" in row["dq_flags"]
    assert "fuzzy_low_confidence" in row["dq_flags"]
    assert row["fault_category"] == "Power"  # fuzzy-cleaned label kept, text says Sensor
    assert "category_text_mismatch" in row["dq_flags"]


def test_logged_categories_pass_through_unchanged_on_the_full_dataset(pipeline_run):
    survivors = pipeline_run.classified.survivors.reset_index(drop=True)
    final = pipeline_run.classified.final
    logged = survivors["fault_category"] != UNKNOWN
    assert (final.loc[logged, "fault_category"] == survivors.loc[logged, "fault_category"]).all()
    assert (final.loc[~logged, "category_source"] != "logged").all()
    assert final["fault_category"].isin([*config.CATEGORIES, UNCLASSIFIED]).all()
    assert len(final) == len(pipeline_run.result.clean)


# --- ML step ---------------------------------------------------------------------------------
PHRASES = {
    "Power": ["battery pack drops voltage", "charging port overheats", "power supply unit fails"],
    "Communications": ["radio loses signal", "antenna connector has weak signal",
                       "network link drops connection"],
    "Calibration": ["calibration unit drifts out of tolerance", "alignment module loses alignment",
                    "reference clock reports an offset error"],
    "Sensor": ["camera module produces blurry image", "thermal sensor is noisy",
               "zoom lens fails self test"],
    "Software": ["ground software crashes on startup", "firmware freezes during upload",
                 "operator ui shows error code"],
    "Mechanical": ["drive belt is cracked", "bearing housing is loose",
                   "servo makes abnormal noise"],
}  # fmt: skip


def _ml_frame(flip: int = 6, blanks: int = 6):
    rng = np.random.default_rng(0)
    rows = [(cat, f"{rng.choice(texts)} {i}") for cat, texts in PHRASES.items() for i in range(30)]
    texts = pd.Series([t for _, t in rows])
    true = pd.Series([c for c, _ in rows])
    logged = true.copy()
    flipped = list(range(0, 180, 180 // flip))[:flip]
    for i in flipped:  # wrong logged label, but the text is clear
        logged[i] = "Power" if true[i] != "Power" else "Sensor"
    blank = list(range(1, 180, 180 // blanks))[:blanks]
    logged[blank] = UNKNOWN
    return texts, logged, true, flipped, blank


def test_ml_does_not_agree_with_a_wrong_logged_label_it_never_saw():
    texts, logged, true, flipped, _ = _ml_frame()
    frame, info, model = ml_predict(texts, logged)
    assert model is not None and info["n_train"] > ML_MIN_TRAIN
    follows_text = (frame.loc[flipped, "ml_category"] == true[flipped]).sum()
    assert follows_text >= len(flipped) - 1  # out-of-fold: text wins over the wrong label


def test_ml_predicts_rows_without_a_logged_category():
    texts, logged, true, _, blank = _ml_frame()
    frame, _, _ = ml_predict(texts, logged)
    assert (frame.loc[blank, "ml_category"] == true[blank]).mean() >= 0.8


def test_ml_abstains_on_text_it_cannot_recognise():
    texts, logged, _, _, _ = _ml_frame()
    frame, _, model = ml_predict(texts, logged)
    from field_ops.classify import ml_apply

    assert ml_apply(model, pd.Series(["zzqx wvv kkpp"])).iloc[0] == UNCLASSIFIED
    assert frame["ml_confidence"].between(0, 1).all()


def test_ml_is_skipped_when_there_is_too_little_labelled_data():
    texts = pd.Series(["battery fails"] * 10)
    frame, info, model = ml_predict(texts, pd.Series(["Power"] * 10))
    assert frame is None and model is None and "skipped" in info


# --- scoring semantics -----------------------------------------------------------------------
def _tiny_truth():
    tickets = pd.DataFrame(
        {
            "report_id": ["FR-1", "FR-2", "FR-3", "FR-4"],
            "description_type": ["clean", "vague", "multi", "mismatch"],
            "fault_category": ["Power", "Sensor", "Calibration", "Software"],
            "secondary_category": ["", "", "Software", ""],
            "logged_category": ["Power", "Sensor", "Calibration", "Power"],
        }
    )
    dups = pd.DataFrame({"report_id": [], "original_report_id": [], "kind": []})
    survivors = pd.DataFrame(
        {"report_id": tickets["report_id"], "fault_category": ["Power", "Sensor", "Calibration",
                                                               "Power"]}
    )  # fmt: skip
    return evaluate.Truth(tickets=tickets, duplicates=dups), survivors


def test_noise_table_counts_guessing_on_vague_text_as_wrong():
    truth, survivors = _tiny_truth()
    pred = pd.Series(["Power", "Sensor", "Software", "Software"])  # vague guessed, multi -> trigger
    table = evaluate.classification_by_noise(survivors, pred, truth).set_index("noise_type")
    assert table.loc["vague text", ["correct", "wrong"]].tolist() == [0, 1]
    multi = table.loc["two categories in one sentence"]
    assert (multi["correct"], multi["wrong"], multi["picked_trigger_category"]) == (0, 1, 1)
    assert table.loc["text contradicts logged category", "correct"] == 1
    assert table.loc["clean text", "accuracy"] == 1.0


def test_noise_table_counts_abstaining_on_vague_text_as_correct():
    truth, survivors = _tiny_truth()
    pred = pd.Series(["Power", UNCLASSIFIED, UNCLASSIFIED, "Software"])
    table = evaluate.classification_by_noise(survivors, pred, truth).set_index("noise_type")
    assert table.loc["vague text", "accuracy"] == 1.0
    assert table.loc["two categories in one sentence", "unclassified"] == 1


def test_contradiction_flag_scores():
    truth, survivors = _tiny_truth()
    pred = pd.Series(["Power", UNCLASSIFIED, "Calibration", "Software"])
    scores = evaluate.mismatch_flag_scores(survivors, pred, truth)
    assert (scores["truly_contradicting"], scores["flagged"], scores["correctly_flagged"]) == (
        1,
        1,
        1,
    )
    assert scores["precision"] == scores["recall"] == 1.0


# --- results on the seeded dataset (regression floors, not real-world accuracy) --------------
def _table(pipeline_run, method):
    return pipeline_run.report.classification["methods"][method]["by_noise"].set_index("noise_type")


def test_rules_never_guess_on_vague_text(pipeline_run):
    row = _table(pipeline_run, "Keyword rules").loc["vague text"]
    assert row["wrong"] == 0
    assert row["accuracy"] >= 0.95


def test_rules_per_noise_type_floors(pipeline_run):
    t = _table(pipeline_run, "Keyword rules")
    assert t.loc["clean text", "accuracy"] >= 0.98
    assert t.loc["two categories in one sentence", "accuracy"] >= 0.90
    assert t.loc["text contradicts logged category", "accuracy"] >= 0.98
    assert not pd.isna(t.loc["macro average (equal weight per type)", "accuracy"])


def test_methods_are_reported_separately(pipeline_run):
    assert list(pipeline_run.report.classification["methods"]) == [
        "Keyword rules",
        "ML (TF-IDF + logistic regression)",
    ]


def test_contradiction_flags_catch_the_planted_contradictions(pipeline_run):
    flags = pipeline_run.report.classification["methods"]["Keyword rules"]["flags"]
    assert flags["recall"] >= 0.95
    assert flags["precision"] >= 0.95
    assert flags["false_flags_by_type"].get("vague text", 0) == 0


def test_blank_categories_are_filled_without_guessing_vague_text(pipeline_run):
    fill = pipeline_run.report.classification["methods"]["Keyword rules"]["fill"]
    assert fill["vague_guessed"] == 0
    assert fill["precision"] >= 0.98
    assert fill["filled"] + fill["unclassified"] == fill["blank_rows"]


# --- challenge set ---------------------------------------------------------------------------
def test_challenge_set_is_well_formed():
    challenge = evaluate.load_challenge()
    assert len(challenge) == 66
    assert set(challenge["expected"]) <= {*config.CATEGORIES, UNCLASSIFIED}
    assert set(challenge["kind"]) == {
        "off-template wording", "vague text", "two categories in one sentence", "misspelled words",
    }  # fmt: skip


def test_challenge_scores_add_up_and_rules_do_not_guess_on_vague(pipeline_run):
    tables = pipeline_run.report.classification["challenge"]
    assert set(tables) == {"Keyword rules", "ML (TF-IDF + logistic regression)"}
    for table in tables.values():
        body = table[table["kind"] != "all hand-written rows"]
        assert (body["correct"] + body["wrong"] + body["unclassified"] == body["rows"]).all()
    rules = tables["Keyword rules"].set_index("kind")
    assert rules.loc["vague text", "accuracy"] == 1.0
    # Off-vocabulary wording must degrade into "unclassified", not into confident wrong answers.
    assert rules.loc["all hand-written rows", "wrong"] <= 3
