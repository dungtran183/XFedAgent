from __future__ import annotations

import gzip

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xfedagent.config import DataConfig
from xfedagent.data import csv_timeseries_dataset
from xfedagent.mimic import (
    NORMAL_VALUES,
    TIMESTEPS,
    _harmonise_units,
    _score_text_values,
    build_cohort_csv,
    build_windows,
    collect_events,
    load_itemid_map,
    select_cohort,
    variable_order,
)

ITEMID_MAP = load_itemid_map()
VARIABLES = variable_order(ITEMID_MAP)
ADMIT = pd.Timestamp("2150-01-01 00:00:00")


def itemid_for(variable: str, linksto: str = "chartevents") -> int:
    rows = ITEMID_MAP[(ITEMID_MAP["variable"] == variable) & (ITEMID_MAP["linksto"] == linksto)]
    assert not rows.empty, f"no {linksto} itemid mapped for {variable!r}"
    return int(rows["itemid"].iloc[0])


def write_mimic(root: Path, stays: list[dict], chartevents: list[dict] = (), labevents: list[dict] = ()) -> Path:
    """A MIMIC-shaped directory holding only the columns the pipeline reads."""
    root.mkdir(parents=True, exist_ok=True)
    icu, patients, admissions = [], [], []
    for stay in stays:
        icu.append(
            {
                "subject_id": stay["subject_id"],
                "hadm_id": stay["hadm_id"],
                "icustay_id": stay["icustay_id"],
                "intime": stay.get("intime", ADMIT),
                "outtime": stay.get("intime", ADMIT) + pd.Timedelta(days=stay["los"]),
                "los": stay["los"],
            }
        )
        patients.append({"subject_id": stay["subject_id"], "gender": "M", "dob": stay["dob"]})
        admissions.append(
            {
                "subject_id": stay["subject_id"],
                "hadm_id": stay["hadm_id"],
                "deathtime": "",
                "hospital_expire_flag": stay.get("expire", 0),
            }
        )
    pd.DataFrame(icu).to_csv(root / "ICUSTAYS.csv", index=False)
    pd.DataFrame(patients).drop_duplicates("subject_id").to_csv(root / "PATIENTS.csv", index=False)
    pd.DataFrame(admissions).drop_duplicates("hadm_id").to_csv(root / "ADMISSIONS.csv", index=False)

    chart_cols = ["subject_id", "hadm_id", "icustay_id", "itemid", "charttime", "value", "valuenum"]
    lab_cols = ["subject_id", "hadm_id", "itemid", "charttime", "value", "valuenum"]
    pd.DataFrame(list(chartevents), columns=chart_cols).to_csv(root / "CHARTEVENTS.csv", index=False)
    pd.DataFrame(list(labevents), columns=lab_cols).to_csv(root / "LABEVENTS.csv", index=False)
    return root


def reading(stay: dict, variable: str, hour: float, value, valuenum=None) -> dict:
    return {
        "subject_id": stay["subject_id"],
        "hadm_id": stay["hadm_id"],
        "icustay_id": stay["icustay_id"],
        "itemid": itemid_for(variable),
        "charttime": stay.get("intime", ADMIT) + pd.Timedelta(hours=hour),
        "value": value,
        "valuenum": value if valuenum is None else valuenum,
    }


ADULT = {"subject_id": 1, "hadm_id": 10, "icustay_id": 100, "los": 6.0, "dob": "2090-01-01"}


# --- cohort selection -------------------------------------------------------


def test_cohort_keeps_only_adults_with_a_stay_long_enough_for_window_and_gap(tmp_path):
    stays = [
        ADULT,
        {"subject_id": 2, "hadm_id": 20, "icustay_id": 200, "los": 6.0, "dob": "2140-01-01"},  # child
        {"subject_id": 3, "hadm_id": 30, "icustay_id": 300, "los": 1.5, "dob": "2090-01-01"},  # 36 h
    ]
    cohort = select_cohort(write_mimic(tmp_path, stays))
    assert list(cohort["icustay_id"]) == [100]


def test_only_the_first_stay_of_an_admission_is_kept(tmp_path):
    """One deterioration must not be counted twice under the same outcome label."""
    stays = [
        {**ADULT, "icustay_id": 101, "intime": ADMIT + pd.Timedelta(days=8)},
        {**ADULT, "icustay_id": 100, "intime": ADMIT},
    ]
    cohort = select_cohort(write_mimic(tmp_path, stays))
    assert list(cohort["icustay_id"]) == [100]


def test_age_of_a_deidentified_over_eighty_nine_patient_does_not_overflow(tmp_path):
    """MIMIC shifts DOB back ~300 years for the very old; a timedelta overflows."""
    stays = [{**ADULT, "dob": "1837-01-01"}]
    cohort = select_cohort(write_mimic(tmp_path, stays))
    assert list(cohort["age"]) == [90]


# --- reading the charted value ---------------------------------------------


def test_capillary_refill_charted_as_text_is_recovered():
    """Regression: the channel is charted as a verdict, so valuenum is empty.

    Dropping those rows made a measured channel indistinguishable from one that
    was never measured, and it read as 0% coverage across the whole cohort.
    """
    events = pd.DataFrame(
        {
            "variable": ["Capillary refill rate"] * 4,
            "value": ["Normal <3 secs", "Abnormal >3 secs", "Brisk", "Delayed"],
            "valuenum": [np.nan] * 4,
        }
    )
    assert list(_score_text_values(events)["valuenum"]) == [0.0, 1.0, 0.0, 1.0]


# The STATUS=ready rows of the itemid selection published with Harutyunyan et al.,
# 114 itemids over the seventeen channels, frozen so an edit to the resource CSV is
# a deliberate departure from the cited protocol. MetaVision's 223951 and 224308 are
# absent because the benchmark records COUNT=0 for both and v1.4 agrees.
PUBLISHED_SELECTION: dict[str, tuple[int, ...]] = {
    "Capillary refill rate": (115, 3348, 8377),
    "Diastolic blood pressure": (8368, 8440, 8441, 8502, 8503, 8504, 8506, 8507, 8555,
                                 220051, 220180, 224643, 225310),
    "Fraction inspired oxygen": (189, 727, 3420, 3422, 223835),
    "Glascow coma scale eye opening": (184, 220739),
    "Glascow coma scale motor response": (454, 223901),
    "Glascow coma scale total": (198,),
    "Glascow coma scale verbal response": (723, 223900),
    "Glucose": (807, 811, 1529, 3745, 50809, 50931, 51478, 220621, 225664, 226537),
    "Heart Rate": (211, 220045),
    "Height": (1394, 226707, 226730),
    "Mean blood pressure": (52, 224, 456, 3312, 3314, 3316, 3320, 3322, 6702,
                            220052, 220181, 224322, 225312),
    "Oxygen saturation": (646, 834, 8498, 50817, 220227, 220277),
    "Respiratory rate": (614, 615, 618, 651, 3603, 220210, 224422, 224689, 224690),
    "Systolic blood pressure": (51, 442, 455, 3313, 3315, 3317, 3321, 3323, 6701,
                                220050, 220179, 224167, 225309, 227243),
    "Temperature": (676, 677, 678, 679, 3654, 3655, 223761, 223762),
    "Weight": (763, 3580, 3581, 3582, 3693, 224639, 226512, 226531),
    "pH": (780, 860, 1126, 1673, 3839, 4202, 4753, 50820, 50831, 51094, 51491,
           220274, 223830),
}


def test_itemid_map_matches_the_published_benchmark_selection():
    """The map is the cited paper's map, itemid for itemid, in both eras."""
    ours = {
        variable: tuple(sorted(int(i) for i in group["itemid"]))
        for variable, group in load_itemid_map().groupby("variable")
    }
    assert ours == {k: tuple(sorted(v)) for k, v in PUBLISHED_SELECTION.items()}
    assert sum(len(v) for v in ours.values()) == 114


def test_gcs_subscore_text_is_recovered():
    events = pd.DataFrame(
        {
            "variable": ["Glascow coma scale eye opening", "Glascow coma scale motor response"],
            "value": ["Spontaneously", "Obeys Commands"],
            "valuenum": [np.nan, np.nan],
        }
    )
    assert list(_score_text_values(events)["valuenum"]) == [4.0, 6.0]


def test_unmapped_free_text_stays_missing_rather_than_guessed():
    """A charting slip such as pH recorded as "." must not become a number."""
    events = pd.DataFrame({"variable": ["pH"], "value": ["."], "valuenum": [np.nan]})
    assert _score_text_values(events)["valuenum"].isna().all()


# --- units and ranges -------------------------------------------------------


@pytest.mark.parametrize(
    "variable,charted,expected",
    [
        ("Temperature", 98.6, 37.0),          # Fahrenheit
        ("Temperature", 37.0, 37.0),          # already Celsius
        ("Fraction inspired oxygen", 40.0, 0.4),
        ("Height", 70.0, 177.8),              # inches
    ],
)
def test_units_are_disambiguated_by_magnitude(variable, charted, expected):
    events = pd.DataFrame({"variable": [variable], "valuenum": [charted]})
    assert _harmonise_units(events)["valuenum"].iloc[0] == pytest.approx(expected, abs=1e-2)


def test_weight_unit_is_taken_from_valueuom_before_magnitude():
    events = pd.DataFrame(
        {"variable": ["Weight", "Weight"], "valuenum": [154.0, 154.0], "valueuom": ["lb", "kg"]}
    )
    values = _harmonise_units(events)["valuenum"].tolist()
    assert values[0] == pytest.approx(69.85, abs=1e-2)
    assert values[1] == pytest.approx(154.0)


def test_weight_itemid_disambiguates_a_blank_declared_unit():
    events = pd.DataFrame(
        {"variable": ["Weight"], "valuenum": [154.0], "valueuom": [""], "itemid": [3581]}
    )
    assert _harmonise_units(events)["valuenum"].iloc[0] == pytest.approx(69.85, abs=1e-2)


def test_ambiguous_blank_weight_is_not_guessed_from_magnitude():
    events = pd.DataFrame({"variable": ["Weight"], "valuenum": [154.0]})
    assert _harmonise_units(events)["valuenum"].iloc[0] == pytest.approx(154.0)


def test_out_of_range_reading_is_discarded_not_clipped():
    """A heart rate of 900 is a charting error; clipping would invent a reading."""
    events = pd.DataFrame({"variable": ["Heart Rate", "Heart Rate"], "valuenum": [80.0, 900.0]})
    kept = _harmonise_units(events)
    assert list(kept["valuenum"]) == [80.0]


# --- windowing --------------------------------------------------------------


def test_readings_in_one_hour_reduce_by_median_so_a_typo_cannot_move_it(tmp_path):
    chart = [reading(ADULT, "Heart Rate", 0.1, v) for v in (80.0, 82.0, 300.0)]
    root = write_mimic(tmp_path, [ADULT], chart)
    cohort = select_cohort(root)
    grid, _ = build_windows(cohort, collect_events(root, cohort, ITEMID_MAP), VARIABLES)
    assert grid[0, 0, VARIABLES.index("Heart Rate")] == pytest.approx(82.0)


def test_a_reading_is_carried_forward_then_falls_back_to_the_normal_value(tmp_path):
    root = write_mimic(tmp_path, [ADULT], [reading(ADULT, "Heart Rate", 5.0, 55.0)])
    cohort = select_cohort(root)
    grid, observed = build_windows(cohort, collect_events(root, cohort, ITEMID_MAP), VARIABLES)
    hr = VARIABLES.index("Heart Rate")
    assert grid[0, 4, hr] == pytest.approx(NORMAL_VALUES["Heart Rate"])  # before any reading
    assert grid[0, 5, hr] == pytest.approx(55.0)
    assert grid[0, 23, hr] == pytest.approx(55.0)                        # carried forward
    assert observed[0, 5, hr] and not observed[0, 6, hr]                 # fill is not observation


def test_readings_after_the_window_are_excluded(tmp_path):
    """Hour 30 sits in the 24-hour gap that keeps the label from leaking in."""
    root = write_mimic(tmp_path, [ADULT], [reading(ADULT, "Heart Rate", 30.0, 55.0)])
    cohort = select_cohort(root)
    _, observed = build_windows(cohort, collect_events(root, cohort, ITEMID_MAP), VARIABLES)
    assert not observed.any()


def test_lab_readings_are_placed_on_the_icu_clock_through_hadm_id(tmp_path):
    """LABEVENTS carries no icustay_id, so it is keyed by admission instead."""
    lab = [
        {
            "subject_id": ADULT["subject_id"],
            "hadm_id": ADULT["hadm_id"],
            "itemid": itemid_for("Glucose", "labevents"),
            "charttime": ADMIT + pd.Timedelta(hours=3),
            "value": "120",
            "valuenum": 120.0,
        }
    ]
    root = write_mimic(tmp_path, [ADULT], [], lab)
    cohort = select_cohort(root)
    grid, observed = build_windows(cohort, collect_events(root, cohort, ITEMID_MAP), VARIABLES)
    glucose = VARIABLES.index("Glucose")
    assert observed[0, 3, glucose]
    assert grid[0, 3, glucose] == pytest.approx(120.0)


# --- the written cohort -----------------------------------------------------


def test_variable_order_is_stable_and_seventeen_channels_wide():
    assert len(VARIABLES) == 17
    assert VARIABLES == sorted(VARIABLES)


def test_cohort_csv_is_timestep_major_and_reshapes_back_the_way_data_py_reshapes(tmp_path):
    """A feature-major header would transpose every sample without erroring."""
    stays = [{**ADULT, "subject_id": i, "hadm_id": 10 * i, "icustay_id": 100 * i} for i in (1, 2, 3, 4)]
    chart = [reading(s, "Heart Rate", 2.0, 70.0 + s["subject_id"]) for s in stays]
    root = write_mimic(tmp_path, stays, chart)
    output = tmp_path / "cohort.csv"
    report = build_cohort_csv(root, output)

    assert report.stays == 4
    assert report.feature_columns == TIMESTEPS * 17 == 408

    frame = pd.read_csv(output)
    features = [c for c in frame.columns if c not in {"subject_id", "mortality_48h"}]
    assert features[:17] == [f"t00_{name}" for name in VARIABLES]
    assert features[-17:] == [f"t23_{name}" for name in VARIABLES]

    block = frame[features].to_numpy().reshape(-1, TIMESTEPS, 17)
    # The builder must keep raw imputed units; fitting a scale here would let
    # validation/test stays influence every later training split.
    assert np.array_equal(frame["t02_Heart Rate"].to_numpy(), [71.0, 72.0, 73.0, 74.0])
    for t in range(TIMESTEPS):
        expected = frame[[f"t{t:02d}_{name}" for name in VARIABLES]].to_numpy()
        assert np.allclose(block[:, t, :], expected)


def test_written_cohort_loads_through_the_configured_csv_reader(tmp_path):
    stays = [
        {**ADULT, "subject_id": i, "hadm_id": 10 * i, "icustay_id": 100 * i, "expire": i % 2}
        for i in range(1, 13)
    ]
    root = write_mimic(tmp_path, stays, [reading(s, "Heart Rate", 1.0, 75.0) for s in stays])
    output = tmp_path / "cohort.csv"
    build_cohort_csv(root, output)

    cfg = DataConfig(
        source="csv_timeseries",
        samples=12,
        clients=2,
        timesteps=TIMESTEPS,
        features=17,
        validation_pool_size=2,
        validation_fraction=0.2,
        test_fraction=0.2,
        dirichlet_alpha=1.0,
        csv_path=str(output),
        label_column="mortality_48h",
        patient_column="subject_id",
    )
    bundle = csv_timeseries_dataset(cfg, seed=0)
    assert bundle.test_x.shape[1:] == (TIMESTEPS, 17)
    assert set(np.unique(bundle.test_y)) <= {0, 1}


def test_cohort_from_the_gzipped_distribution_is_byte_identical(tmp_path):
    """PhysioNet serves the full release gzipped, the demo plain.

    The reported cohort must not depend on which of the two the tables arrived
    as, so the two builds are compared byte for byte rather than row by row.
    """
    stays = [{**ADULT, "subject_id": i, "hadm_id": 10 * i, "icustay_id": 100 * i} for i in (1, 2, 3)]
    chart = [reading(s, "Heart Rate", 2.0, 70.0 + s["subject_id"]) for s in stays]

    plain = write_mimic(tmp_path / "plain", stays, chart)
    gzipped = tmp_path / "gz"
    gzipped.mkdir()
    for table in ("PATIENTS", "ADMISSIONS", "ICUSTAYS", "CHARTEVENTS", "LABEVENTS"):
        with gzip.open(gzipped / f"{table}.csv.gz", "wb") as handle:
            handle.write((plain / f"{table}.csv").read_bytes())

    from_plain, from_gzipped = tmp_path / "a.csv", tmp_path / "b.csv"
    plain_report = build_cohort_csv(plain, from_plain)
    gzipped_report = build_cohort_csv(gzipped, from_gzipped)

    assert plain_report.stays == gzipped_report.stays == 3
    assert from_plain.read_bytes() == from_gzipped.read_bytes()
