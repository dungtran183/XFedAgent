from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xfedagent.challenge2012 import (
    PARAMETER_TO_VARIABLE,
    build_cohort_csv,
    drop_out_of_range,
    parse_record,
    read_outcomes,
    scan_set,
    select_cohort,
)
from xfedagent.config import DataConfig
from xfedagent.data import csv_timeseries_dataset
from xfedagent.mimic import TIMESTEPS, load_itemid_map, variable_order

VARIABLES = variable_order(load_itemid_map())

# The four channels Challenge 2012 does not publish. Only the GCS total is given,
# not its three sub-scores, and capillary refill is not charted at all.
CHALLENGE_ABSENT = (
    "Capillary refill rate",
    "Glascow coma scale eye opening",
    "Glascow coma scale motor response",
    "Glascow coma scale verbal response",
)
SET_A = Path("data/raw/challenge-2012")


def write_record(set_dir: Path, record_id: int, rows: list[tuple[str, str, float]], age: float = 70.0) -> Path:
    """One Challenge record file: a header, the descriptors, then the readings."""
    set_dir.mkdir(parents=True, exist_ok=True)
    path = set_dir / f"{record_id}.txt"
    lines = ["Time,Parameter,Value", f"00:00,RecordID,{record_id}", f"00:00,Age,{age}",
             "00:00,Gender,1", "00:00,ICUType,4"]
    lines += [f"{t},{p},{v}" for t, p, v in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_outcomes(root: Path, records: list[tuple[int, float, int]], set_letter: str = "a") -> Path:
    path = root / f"Outcomes-{set_letter}.txt"
    frame = pd.DataFrame(
        [
            {
                "RecordID": rid,
                "SAPS-I": 6,
                "SOFA": 1,
                "Length_of_stay": los,
                "Survival": -1,
                "In-hospital_death": death,
            }
            for rid, los, death in records
        ]
    )
    frame.to_csv(path, index=False)
    return path


def test_parse_record_keeps_the_window_and_drops_the_gap(tmp_path):
    path = write_record(
        tmp_path / "set-a",
        132539,
        [("00:15", "HR", 71.0), ("23:59", "HR", 80.0), ("24:00", "HR", 90.0), ("47:00", "HR", 95.0)],
    )
    descriptors, rows = parse_record(path)
    assert descriptors["Age"] == 70.0
    assert descriptors["_after_window"] == 2.0
    assert [(hour, variable) for _, hour, variable, _ in rows] == [(0, "Heart Rate"), (23, "Heart Rate")]


def test_parse_record_ignores_parameters_outside_the_seventeen_channels(tmp_path):
    path = write_record(
        tmp_path / "set-a",
        1,
        [("01:00", "Urine", 200.0), ("01:00", "Lactate", 2.1), ("01:00", "HR", 70.0)],
    )
    _, rows = parse_record(path)
    assert [variable for _, _, variable, _ in rows] == ["Heart Rate"]


def test_parse_record_rejects_a_file_that_is_not_challenge_shaped(tmp_path):
    path = tmp_path / "142673.txt"
    path.write_text("time,param,value\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected header"):
        parse_record(path)


def test_parse_record_rejects_a_file_whose_name_is_not_a_record_id(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Time,Parameter,Value\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a Challenge record id"):
        parse_record(path)


def test_height_and_weight_are_readings_not_descriptors(tmp_path):
    """Both are charted at 00:00, and both are among the seventeen channels."""
    path = write_record(tmp_path / "set-a", 2, [("00:00", "Weight", 80.0), ("00:00", "Height", 170.0)])
    descriptors, rows = parse_record(path)
    assert "Weight" not in descriptors and "Height" not in descriptors
    assert sorted(variable for _, _, variable, _ in rows) == ["Height", "Weight"]


def test_read_outcomes_drops_a_record_with_no_published_stay_length(tmp_path):
    write_outcomes(tmp_path, [(1, 5.0, 0), (2, -1.0, 1), (3, 3.0, 1)])
    outcomes = read_outcomes(tmp_path / "Outcomes-a.txt")
    assert list(outcomes["icustay_id"]) == [1, 3]
    assert list(outcomes["in_hospital_death"]) == [0, 1]


def test_read_outcomes_rejects_a_table_missing_the_label(tmp_path):
    path = tmp_path / "Outcomes-a.txt"
    pd.DataFrame([{"RecordID": 1, "Length_of_stay": 5}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="In-hospital_death"):
        read_outcomes(path)


def test_select_cohort_applies_the_adult_and_survival_filters(tmp_path):
    set_dir = tmp_path / "set-a"
    for rid, age in ((1, 70.0), (2, 15.0), (3, 45.0), (4, 80.0)):
        write_record(set_dir, rid, [("01:00", "HR", 70.0)], age=age)
    write_outcomes(tmp_path, [(1, 5.0, 0), (2, 6.0, 1), (3, 2.0, 1), (4, 9.0, 1)])
    descriptors, _, _ = scan_set(set_dir)
    cohort = select_cohort(descriptors, read_outcomes(tmp_path / "Outcomes-a.txt"))
    # 2 is a child, 3 does not outlast the window plus the gap.
    assert list(cohort["icustay_id"]) == [1, 4]
    assert list(cohort["subject_id"]) == [1, 4]


def test_out_of_range_readings_are_discarded_not_clipped():
    readings = pd.DataFrame(
        [
            {"icustay_id": 1, "hour": 0, "variable": "pH", "valuenum": 735.0},
            {"icustay_id": 1, "hour": 0, "variable": "pH", "valuenum": 7.4},
            {"icustay_id": 1, "hour": 0, "variable": "Height", "valuenum": -1.0},
            {"icustay_id": 1, "hour": 1, "variable": "Temperature", "valuenum": 0.8},
        ]
    )
    kept, dropped = drop_out_of_range(readings)
    assert dropped == 3
    assert list(kept["valuenum"]) == [7.4]


def test_invasive_and_non_invasive_pressure_share_one_channel(tmp_path):
    """Two ways of charting the same quantity must not become two channels."""
    assert PARAMETER_TO_VARIABLE["SysABP"] == PARAMETER_TO_VARIABLE["NISysABP"]
    path = write_record(
        tmp_path / "set-a", 5, [("03:10", "SysABP", 110.0), ("03:40", "NISysABP", 130.0)]
    )
    _, rows = parse_record(path)
    assert {variable for _, _, variable, _ in rows} == {"Systolic blood pressure"}
    # build_windows medians within the hour, so the stay contributes 120, once.
    frame = pd.DataFrame(rows, columns=["icustay_id", "hour", "variable", "valuenum"])
    hourly = frame.groupby(["icustay_id", "hour", "variable"], as_index=False)["valuenum"].median()
    assert len(hourly) == 1 and hourly["valuenum"].iloc[0] == 120.0


def test_build_cohort_csv_writes_the_schema_the_loader_reads(tmp_path):
    set_dir = tmp_path / "set-a"
    for rid in (1, 2, 3, 4, 5, 6):
        write_record(
            set_dir,
            rid,
            [(f"{h:02d}:30", "HR", 70.0 + rid + h) for h in range(TIMESTEPS)]
            + [("05:00", "Temp", 37.0), ("30:00", "HR", 999.0)],
        )
    write_outcomes(tmp_path, [(rid, 6.0, rid % 2) for rid in (1, 2, 3, 4, 5, 6)])

    output = tmp_path / "cohort.csv"
    report = build_cohort_csv(tmp_path, output)

    assert report.cohort.stays == 6
    assert report.cohort.patients == 6
    assert report.cohort.positives == 3
    assert report.cohort.feature_columns == TIMESTEPS * len(VARIABLES) == 408
    assert report.records_scanned == 6
    assert report.readings_after_window == 6  # the 30:00 reading in each record

    frame = pd.read_csv(output)
    assert frame.columns[0] == "subject_id"
    assert frame.columns[-1] == "in_hospital_death"
    assert list(frame.columns[1:4]) == [f"t00_{name}" for name in VARIABLES[:3]]
    # Timestep-major: hour 1 begins only after all seventeen channels of hour 0.
    assert frame.columns[1 + len(VARIABLES)] == f"t01_{VARIABLES[0]}"


def test_the_four_channels_the_challenge_does_not_publish_are_named_and_constant(tmp_path):
    set_dir = tmp_path / "set-a"
    for rid in (1, 2, 3):
        write_record(set_dir, rid, [("02:00", "HR", 70.0 + rid), ("02:00", "GCS", 14.0)])
    write_outcomes(tmp_path, [(rid, 6.0, 0) for rid in (1, 2, 3)])

    output = tmp_path / "cohort.csv"
    report = build_cohort_csv(tmp_path, output)
    assert set(CHALLENGE_ABSENT) <= set(report.absent_variables)
    # The GCS total is published, so it is never among the absent channels even
    # though its three sub-scores always are.
    assert "Glascow coma scale total" not in report.absent_variables
    # Absent means imputed at the normal value for every stay and every hour, so
    # it remains constant in the clinical-unit CSV. The train-only loader will
    # map that constant to zero after the split.
    frame = pd.read_csv(output)
    for name in report.absent_variables:
        columns = [f"t{t:02d}_{name}" for t in range(TIMESTEPS)]
        from xfedagent.mimic import NORMAL_VALUES
        assert np.allclose(frame[columns].to_numpy(), NORMAL_VALUES[name]), name


def test_the_cohort_loads_through_the_existing_csv_timeseries_path(tmp_path):
    set_dir = tmp_path / "set-a"
    for rid in range(1, 25):
        write_record(set_dir, rid, [(f"{h:02d}:00", "HR", 60.0 + rid + h) for h in range(TIMESTEPS)])
    write_outcomes(tmp_path, [(rid, 6.0, rid % 2) for rid in range(1, 25)])
    output = tmp_path / "cohort.csv"
    build_cohort_csv(tmp_path, output)

    cfg = DataConfig(
        source="csv_timeseries",
        samples=24,
        clients=2,
        timesteps=TIMESTEPS,
        features=len(VARIABLES),
        validation_pool_size=4,
        validation_fraction=0.2,
        test_fraction=0.2,
        dirichlet_alpha=0.5,
        csv_path=str(output),
        label_column="in_hospital_death",
        patient_column="subject_id",
        positive_rate=0.5,
    )
    bundle = csv_timeseries_dataset(cfg, seed=0)
    assert len(bundle.clients) == 2
    assert bundle.validation_pool_x.shape[1:] == (TIMESTEPS, len(VARIABLES))
    assert bundle.test_x.shape[1:] == (TIMESTEPS, len(VARIABLES))


def test_a_cohort_with_no_eligible_record_is_refused(tmp_path):
    write_record(tmp_path / "set-a", 1, [("01:00", "HR", 70.0)], age=10.0)
    write_outcomes(tmp_path, [(1, 6.0, 0)])
    with pytest.raises(ValueError, match="adult"):
        build_cohort_csv(tmp_path, tmp_path / "cohort.csv")


@pytest.mark.skipif(
    not (SET_A / "set-a").is_dir() or not (SET_A / "Outcomes-a.txt").is_file(),
    reason="Challenge 2012 set A not downloaded; data/ is gitignored",
)
def test_real_set_a_matches_what_the_module_documents(tmp_path):
    """Pin the claims the docstring makes about the published data itself.

    Everything else here runs on a fixture, which cannot catch the Challenge
    changing shape or the mapping drifting away from the 17 channels. This one
    reads set A as distributed. It skips rather than fails where the data is not
    present, because ``data/`` is not part of the repository.
    """
    report = build_cohort_csv(SET_A, tmp_path / "cohort.csv")
    assert report.records_scanned == 4000
    assert report.records_without_outcome == 60  # Length_of_stay is -1 for these
    assert report.cohort.stays == 3883
    assert report.cohort.patients == report.cohort.stays  # no published linkage
    assert report.cohort.feature_columns == 408
    # 13.75%, within the 1-point tolerance preflight applies to the MIMIC cohort's
    # 14.0%, so this cohort can rehearse the MIMIC configuration.
    assert report.cohort.prevalence == pytest.approx(0.1375, abs=0.001)
    assert report.absent_variables == CHALLENGE_ABSENT
