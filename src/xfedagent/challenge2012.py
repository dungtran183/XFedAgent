"""PhysioNet/CinC Challenge 2012 to the same flattened cohort schema.

This cohort exists to check the pipeline and the paired-predicate comparison end
to end while MIMIC-III access is still pending, and for nothing else. It is an
internal cross-check: no number produced from it is reported in the manuscript,
none of it is fed to ``xfedagent macros``, and it is not a substitute for the
MIMIC-III cohort the paper describes. Set A of the Challenge is open access
(no credentialing, no DUA), which is the only reason it can be used here.

Why it is a usable stand-in for a pipeline check. The Challenge records cover
the first 48 h of an adult ICU stay, so the manuscript's design -- a 24-hour
observation window followed by a 24-hour gap, with the outcome read from the
hospital record -- fits without altering it: hours 0-23 are the window and
hours 24-47 are dropped as the gap. Restricting to stays longer than 48 h puts
in-hospital-death prevalence at 13.7%, the same figure the MIMIC cohort is
built to.

Where it differs, and this is why it cannot stand in for the reported cohort:
Challenge 2012 publishes 13 of the 17 channels. Capillary refill rate and the
three Glasgow Coma Scale sub-scores are absent -- only the GCS total is given --
so those four channels carry their ``NORMAL_VALUES`` constant for every stay and
z-score to zero. Thirteen live channels out of seventeen is enough to exercise
the gate, the predicate and the aggregation; it is not the input the manuscript
reports on.

Units are already harmonised by the Challenge distribution, which publishes one
unit per parameter (Temp in Celsius, FiO2 as a fraction, Weight in kg, Height in
cm), so the Fahrenheit/pound/percent reconciliation ``mimic.py`` needs has no
counterpart here. Verified against set A rather than assumed: FiO2 spans
0.21-1.00, Temp has median 37.1, Weight median 80.7, Height median 170.2. What
does survive is the range filter, because the Challenge carries charting errors
of its own -- a pH of 735, a height of 431.8 cm -- and ``VALID_RANGES`` discards
those rather than clipping them, exactly as the MIMIC path does.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd

from .mimic import (
    TIMESTEPS,
    VALID_RANGES,
    CohortReport,
    build_windows,
    load_itemid_map,
    variable_order,
)

# Challenge parameter name -> one of the 17 manuscript channels. Invasive and
# non-invasive pressures merge into a single channel, the way several MIMIC
# itemids merge into one variable: build_windows takes the median within an
# hour, so a stay charted both ways contributes one value per hour rather than
# two competing ones.
PARAMETER_TO_VARIABLE: dict[str, str] = {
    "HR": "Heart Rate",
    "Temp": "Temperature",
    "RespRate": "Respiratory rate",
    "GCS": "Glascow coma scale total",
    "FiO2": "Fraction inspired oxygen",
    "Glucose": "Glucose",
    "pH": "pH",
    "SaO2": "Oxygen saturation",
    "Weight": "Weight",
    "Height": "Height",
    "SysABP": "Systolic blood pressure",
    "NISysABP": "Systolic blood pressure",
    "DiasABP": "Diastolic blood pressure",
    "NIDiasABP": "Diastolic blood pressure",
    "MAP": "Mean blood pressure",
    "NIMAP": "Mean blood pressure",
}

# Descriptors recorded once at 00:00 that are not channel readings. Height and
# Weight are deliberately absent from this set: they are charted at 00:00 too,
# but they are two of the seventeen channels, so they are kept as readings.
DESCRIPTOR_PARAMETERS = frozenset({"RecordID", "Age", "Gender", "ICUType"})

# The Challenge writes -1 for a descriptor it does not have. Every entry in
# VALID_RANGES has a lower bound of at least 0.0, so the range filter already
# discards those for channel readings; only Age is read outside it and needs the
# sentinel handled explicitly.
MISSING_SENTINEL = -1.0


@dataclass(frozen=True)
class Challenge2012Report:
    """The cohort figures plus what the Challenge could not supply."""

    cohort: CohortReport
    absent_variables: tuple[str, ...]
    records_scanned: int
    records_without_outcome: int
    readings_out_of_range: int
    readings_after_window: int

    def to_dict(self) -> dict[str, float | int | str]:
        merged: dict[str, float | int | str] = dict(self.cohort.to_dict())
        merged["absent_variables"] = ";".join(self.absent_variables)
        merged["records_scanned"] = self.records_scanned
        merged["records_without_outcome"] = self.records_without_outcome
        merged["readings_out_of_range"] = self.readings_out_of_range
        merged["readings_after_window"] = self.readings_after_window
        return merged


def read_outcomes(path: Path | str) -> pd.DataFrame:
    """The published outcome table for one set.

    ``Length_of_stay`` is in days and is -1 where the Challenge does not have it;
    such a record cannot be placed on either side of the >48 h filter, so it is
    dropped rather than guessed at.
    """
    frame = pd.read_csv(path)
    required = {"RecordID", "Length_of_stay", "In-hospital_death"}
    if not required <= set(frame.columns):
        raise ValueError(f"{path} is missing {sorted(required - set(frame.columns))}")
    frame = frame.rename(
        columns={
            "RecordID": "icustay_id",
            "Length_of_stay": "los",
            "In-hospital_death": "in_hospital_death",
        }
    )
    frame = frame[["icustay_id", "los", "in_hospital_death"]].astype(
        {"icustay_id": "int64", "los": "float64", "in_hospital_death": "int64"}
    )
    return frame[frame["los"] != MISSING_SENTINEL].reset_index(drop=True)


def parse_record(path: Path | str) -> tuple[dict[str, float], list[tuple[int, int, str, float]]]:
    """One ``set-a/<RecordID>.txt`` file into descriptors and channel readings.

    Returned rows are ``(record_id, hour, variable, value)`` for the observation
    window only. Readings from hour 24 onward are the gap and are discarded here,
    at the parse, so nothing downstream can see them.
    """
    path = Path(path)
    # The file name is the record id; the Challenge publishes no other copy of
    # it inside the file that can be trusted before the header is checked.
    try:
        record_id = int(path.stem)
    except ValueError as error:
        raise ValueError(f"{path}: file name is not a Challenge record id") from error
    descriptors: dict[str, float] = {}
    rows: list[tuple[int, int, str, float]] = []
    after_window = 0
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        if header[:3] != ["Time", "Parameter", "Value"]:
            raise ValueError(f"{path}: unexpected header {header!r}")
        for line in handle:
            line = line.strip()
            if not line:
                continue
            timestamp, parameter, raw = line.split(",")
            if parameter in DESCRIPTOR_PARAMETERS:
                descriptors[parameter] = float(raw)
                continue
            variable = PARAMETER_TO_VARIABLE.get(parameter)
            if variable is None:
                continue
            hour = int(timestamp.split(":")[0])
            if hour >= TIMESTEPS:
                after_window += 1
                continue
            rows.append((record_id, hour, variable, float(raw)))
    descriptors["_record_id"] = float(record_id)
    descriptors["_after_window"] = float(after_window)
    return descriptors, rows


def scan_set(set_dir: Path | str) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Read every record in a set once, returning descriptors and readings."""
    set_dir = Path(set_dir)
    files = sorted(set_dir.glob("*.txt"), key=lambda p: int(p.stem))
    if not files:
        raise ValueError(f"no Challenge record files in {set_dir}")
    descriptor_rows: list[dict[str, float]] = []
    reading_rows: list[tuple[int, int, str, float]] = []
    after_window = 0
    for path in files:
        descriptors, rows = parse_record(path)
        after_window += int(descriptors.pop("_after_window"))
        descriptor_rows.append(descriptors)
        reading_rows.extend(rows)
    descriptors_frame = pd.DataFrame(descriptor_rows).rename(columns={"_record_id": "icustay_id"})
    descriptors_frame["icustay_id"] = descriptors_frame["icustay_id"].astype("int64")
    readings = pd.DataFrame(
        reading_rows, columns=["icustay_id", "hour", "variable", "valuenum"]
    )
    return descriptors_frame, readings, after_window


def select_cohort(
    descriptors: pd.DataFrame,
    outcomes: pd.DataFrame,
    min_los_days: float = 2.0,
    min_age: int = 18,
) -> pd.DataFrame:
    """Adults whose stay outlasts the window and the gap.

    The same two filters ``mimic.select_cohort`` applies, for the same reason:
    the label is only meaningful once the stay reaches the end of the gap. Each
    Challenge record is one ICU stay, and the Challenge publishes no linkage
    between records, so the record id serves as both stay and patient identifier
    -- which makes the downstream patient-disjoint split trivially satisfied.
    """
    cohort = descriptors.merge(outcomes, on="icustay_id", how="inner")
    age = cohort.get("Age")
    if age is None:
        raise ValueError("no Age descriptor found; is this a Challenge 2012 set?")
    cohort = cohort[(age >= min_age) & (cohort["los"] > min_los_days)]
    cohort = cohort.sort_values("icustay_id").reset_index(drop=True)
    cohort["subject_id"] = cohort["icustay_id"]
    return cohort


def drop_out_of_range(readings: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Discard physiologically impossible readings, and count them.

    Out of range is discarded, never clipped: the Challenge contains a pH of 735
    and a height of 431.8 cm, and clipping would turn each into a plausible
    extreme instead of the charting error it is. The -1 the Challenge writes for
    a missing descriptor also lands here, because every valid range starts at 0
    or above.
    """
    if readings.empty:
        return readings, 0
    low = readings["variable"].map(lambda v: VALID_RANGES[v][0])
    high = readings["variable"].map(lambda v: VALID_RANGES[v][1])
    keep = (readings["valuenum"] >= low) & (readings["valuenum"] <= high)
    return readings[keep].reset_index(drop=True), int((~keep).sum())


def build_cohort_csv(
    challenge_dir: Path | str,
    output: Path | str = "data/processed/challenge2012_mortality_24h.csv",
    set_name: str = "set-a",
    label_column: str = "in_hospital_death",
) -> Challenge2012Report:
    """Run the whole pipeline and write the flattened cohort.

    The output is the schema ``data.py`` already reads -- ``subject_id``, 24x17
    timestep-major feature columns, then the label -- so the Challenge cohort
    drops into the existing ``csv_timeseries`` loader with no change to it. The
    label column is named for what it is rather than reusing the MIMIC path's
    ``mortality_48h``, so a cohort file cannot be mistaken for the other.
    """
    challenge_dir = Path(challenge_dir)
    output = Path(output)
    variables = variable_order(load_itemid_map())

    outcomes = read_outcomes(challenge_dir / f"Outcomes-{set_name.split('-')[-1]}.txt")
    descriptors, readings, after_window = scan_set(challenge_dir / set_name)
    cohort = select_cohort(descriptors, outcomes)
    if cohort.empty:
        raise ValueError("no Challenge record survived the adult / >48 h filters")

    readings = readings[readings["icustay_id"].isin(set(cohort["icustay_id"]))]
    readings, out_of_range = drop_out_of_range(readings)
    observed_variables = set(readings["variable"].unique())
    absent = tuple(name for name in variables if name not in observed_variables)

    grid, observed = build_windows(cohort, readings, variables)

    columns = [f"t{t:02d}_{name}" for t in range(TIMESTEPS) for name in variables]
    # Each seed's train-only standardisation is fitted by the CSV loader.
    flat = grid.reshape(len(cohort), TIMESTEPS * len(variables))
    frame = pd.DataFrame(flat, columns=columns)
    frame.insert(0, "subject_id", cohort["subject_id"].to_numpy())
    frame[label_column] = cohort["in_hospital_death"].to_numpy().astype(int)

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    positives = int(frame[label_column].sum())
    return Challenge2012Report(
        cohort=CohortReport(
            stays=len(frame),
            patients=int(frame["subject_id"].nunique()),
            positives=positives,
            prevalence=positives / len(frame),
            feature_columns=len(columns),
            imputed_fraction=float(1.0 - observed.mean()),
            output=str(output),
        ),
        absent_variables=absent,
        records_scanned=len(descriptors),
        records_without_outcome=len(descriptors) - len(descriptors.merge(outcomes, on="icustay_id")),
        readings_out_of_range=out_of_range,
        readings_after_window=after_window,
    )


if __name__ == "__main__":  # pragma: no cover
    report = build_cohort_csv(
        sys.argv[1],
        sys.argv[2] if len(sys.argv) > 2 else "data/processed/challenge2012_mortality_24h.csv",
    )
    for key, value in report.to_dict().items():
        print(f"{key}={value}")
