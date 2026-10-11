import numpy as np
import pandas as pd
import pytest

from field_ops import config
from field_ops.generate import Scenario, describe, generate_dataset, write_dataset
from field_ops.mess import RAW_COLUMNS, typo


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset()


def test_raw_has_only_business_columns(dataset):
    assert list(dataset.raw.columns) == RAW_COLUMNS


def test_truth_never_written_to_raw_dir(dataset, tmp_path):
    raw_path = tmp_path / "raw" / "fault_reports.csv"
    write_dataset(dataset, raw_path=raw_path, truth_dir=tmp_path / "truth")
    assert [p.name for p in raw_path.parent.iterdir()] == ["fault_reports.csv"]
    assert list(pd.read_csv(raw_path).columns) == RAW_COLUMNS


def test_same_seed_is_reproducible(dataset):
    again = generate_dataset()
    pd.testing.assert_frame_equal(dataset.raw, again.raw)


def test_different_seed_changes_data(dataset):
    other = generate_dataset(Scenario(seed=7))
    assert not dataset.raw["description"].equals(other.raw["description"])


def test_size_is_about_two_thousand(dataset):
    assert 1900 <= len(dataset.raw) <= 2150


def test_duplicate_answer_key_is_consistent(dataset):
    raw_ids = dataset.raw["report_id"]
    truth_ids = set(dataset.truth["report_id"])
    dups = dataset.duplicates

    exact = dups[dups["kind"] == "exact"]
    assert (raw_ids.value_counts()[exact["report_id"]] >= 2).all()

    near = dups[dups["kind"] == "near"]
    assert near["original_report_id"].isin(truth_ids).all()
    assert not near["report_id"].isin(truth_ids).any()
    # Every raw ID is either a real ticket or a known near-duplicate.
    assert set(raw_ids) == truth_ids | set(near["report_id"])


def test_description_noise_rates(dataset):
    share = dataset.truth["description_type"].value_counts(normalize=True)
    assert share["vague"] == pytest.approx(0.08, abs=0.02)
    assert share["multi"] == pytest.approx(0.07, abs=0.02)
    assert share["mismatch"] == pytest.approx(0.05, abs=0.015)
    mismatch = dataset.truth[dataset.truth["description_type"] == "mismatch"]
    assert (mismatch["logged_category"] != mismatch["fault_category"]).all()


def test_closed_tickets_close_after_opening_in_truth(dataset):
    t = dataset.truth
    closed = t[t["status"] == "Closed"]
    assert closed["date_closed"].notna().all()
    assert (closed["date_closed"] >= closed["date_opened"]).all()
    assert t.loc[t["status"] != "Closed", "date_closed"].isna().all()


def test_planted_slow_vendors_are_detectable_by_median(dataset):
    t = dataset.truth
    closed = t[t["status"] == "Closed"]
    days = closed.apply(lambda r: (r["date_closed"] - r["date_opened"]).days, axis=1)
    medians = days.groupby(closed["vendor"]).median()
    slow = [v for v, _ in Scenario().slow_vendors]
    assert medians[slow].min() > medians.drop(slow).median()


def test_lemon_systems_have_highest_repeat_rates(dataset):
    rate = dataset.truth.groupby("system_id")["is_recurrence"].mean()
    assert set(rate.nlargest(5).index) == set(Scenario().lemon_systems)


def test_describe_multi_mentions_both_categories():
    rng = np.random.default_rng(0)
    text = describe(rng, "Calibration", "multi", secondary="Software")
    assert any(s.lower() in text.lower() for s in ("firmware", "software"))
    assert any(w in text.lower() for w in ("calibration", "alignment", "reference clock", "timing"))


def test_typo_changes_one_word_only():
    rng = np.random.default_rng(1)
    original = "Battery pack overheats while charging"
    changed = typo(rng, original)
    diff = [a != b for a, b in zip(original.split(), changed.split(), strict=True)]
    assert sum(diff) == 1


def test_dates_inside_reporting_window(dataset):
    opened = pd.to_datetime(dataset.truth["date_opened"])
    assert opened.min().date() >= config.START_DATE
    assert opened.max().date() <= config.END_DATE
