"""MIMIC-III to the flattened cohort the experiment configs expect.

Implements the preprocessing of Harutyunyan et al. as the manuscript specifies
it: adult ICU stays longer than 48 h, 17 hourly vitals and labs, forward-fill
imputation, a 24-hour observation window followed by a
24-hour gap, and an in-hospital mortality label.

The window is 24 h rather than the benchmark's 48 h because of the gap: hours
24-48 cover the period over which the outcome is partly already determined, so
they are dropped and the label is read from the hospital record. The 48 h stay
requirement is what makes the window and the gap both fit, and it puts cohort
prevalence at the 14.0% the manuscript reports.

No patient-level data is written anywhere but the output CSV, and ``data/`` is
git-ignored. The itemid map in ``resources/`` is derived from the MIT-licensed
mimic3-benchmarks repository (YerevaNN), trimmed to the 17 variables it marks
``ready``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

RESOURCE_MAP = Path(__file__).parent / "resources" / "itemid_to_variable_map.csv"

# Timestep-major, matching the reshape to (timesteps, features) in data.py.
TIMESTEPS = 24
GAP_HOURS = 24

# Benchmark healthy-adult values, used when a channel has no observation at or
# before an hour.
NORMAL_VALUES: dict[str, float] = {
    "Capillary refill rate": 0.0,
    "Diastolic blood pressure": 59.0,
    "Fraction inspired oxygen": 0.21,
    "Glascow coma scale eye opening": 4.0,
    "Glascow coma scale motor response": 6.0,
    "Glascow coma scale total": 15.0,
    "Glascow coma scale verbal response": 5.0,
    "Glucose": 128.0,
    "Heart Rate": 86.0,
    "Height": 170.0,
    "Mean blood pressure": 77.0,
    "Oxygen saturation": 98.0,
    "Respiratory rate": 19.0,
    "Systolic blood pressure": 118.0,
    "Temperature": 36.6,
    "Weight": 81.0,
    "pH": 7.4,
}

# Physiologically possible ranges. Readings outside their range are discarded
# before imputation, not clipped onto the boundary.
VALID_RANGES: dict[str, tuple[float, float]] = {
    "Capillary refill rate": (0.0, 1.0),
    "Diastolic blood pressure": (0.0, 200.0),
    "Fraction inspired oxygen": (0.2, 1.0),
    "Glascow coma scale eye opening": (1.0, 4.0),
    "Glascow coma scale motor response": (1.0, 6.0),
    "Glascow coma scale total": (3.0, 15.0),
    "Glascow coma scale verbal response": (1.0, 5.0),
    "Glucose": (0.0, 2000.0),
    "Heart Rate": (0.0, 350.0),
    "Height": (30.0, 260.0),
    "Mean blood pressure": (0.0, 250.0),
    "Oxygen saturation": (0.0, 100.0),
    "Respiratory rate": (0.0, 120.0),
    "Systolic blood pressure": (0.0, 300.0),
    "Temperature": (14.0, 47.0),
    "Weight": (10.0, 500.0),
    "pH": (6.3, 8.0),
}

# Glasgow Coma Scale sub-scores are charted as text in both source systems, with
# differing spellings. Both are listed.
GCS_TEXT_TO_SCORE: dict[str, float] = {
    # eye opening
    "none": 1.0, "no response": 1.0,
    "to pain": 2.0, "to painful stimuli": 2.0,
    "to speech": 3.0, "to voice": 3.0, "to sound": 3.0,
    "spontaneously": 4.0,
    # motor response
    "no response-ett": 1.0,
    "abnormal extension": 2.0, "abnorm extensn": 2.0,
    "abnormal flexion": 3.0, "abnorm flexion": 3.0,
    "flex-withdraws": 4.0, "flexion-withdrawal": 4.0, "withdraws": 4.0,
    "localizes pain": 5.0, "localizes to pain": 5.0,
    "obeys commands": 6.0,
    # verbal response
    "incomprehensible sounds": 2.0, "incomprehensible": 2.0,
    "inappropriate words": 3.0,
    "confused": 4.0,
    "oriented": 5.0,
}

GCS_VARIABLES = frozenset(
    {
        "Glascow coma scale eye opening",
        "Glascow coma scale motor response",
        "Glascow coma scale verbal response",
    }
)

# Charted as a verdict rather than a number, so the labels map onto the binary
# 0/1 of VALID_RANGES. Both CareVue spellings are listed. Of the three CareVue
# itemids, 3348 is the neonatal one and drops out under ``age >= 18``, leaving
# 115 and 8377 on 795 cohort stays. MetaVision 223951/224308 record COUNT=0 in
# the published selection and are not mapped.
CAPILLARY_TEXT_TO_SCORE: dict[str, float] = {
    "normal <3 secs": 0.0,
    "normal": 0.0,
    "brisk": 0.0,
    "abnormal >3 secs": 1.0,
    "abnormal": 1.0,
    "delayed": 1.0,
}

# ``valueuom`` is read first but is blank for a large minority of readings, so
# magnitude is a fallback rather than the rule: 154 is plausible in either
# pounds or kilograms.
POUND_UNITS = frozenset({"lb", "lbs", "lb.", "lbs.", "pound", "pounds"})
OUNCE_UNITS = frozenset({"oz", "oz.", "ounce", "ounces"})
INCH_UNITS = frozenset({"in", "in.", "inch", "inches"})

# Weight concepts whose unit is fixed in D_ITEMS, consulted before the
# magnitude fallback.
POUND_WEIGHT_ITEMIDS = frozenset({3581, 226531})
OUNCE_WEIGHT_ITEMIDS = frozenset({3582})

CAPILLARY_VARIABLE = "Capillary refill rate"

# Channels carrying their measurement in ``value`` rather than ``valuenum``.
# Anything outside this table reads the numeric column only.
TEXT_TO_SCORE: dict[str, dict[str, float]] = {
    **{variable: GCS_TEXT_TO_SCORE for variable in GCS_VARIABLES},
    CAPILLARY_VARIABLE: CAPILLARY_TEXT_TO_SCORE,
}


@dataclass(frozen=True)
class CohortReport:
    """What the cohort looks like, so the caller can check it before training."""

    stays: int
    patients: int
    positives: int
    prevalence: float
    feature_columns: int
    imputed_fraction: float
    output: str

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "stays": self.stays,
            "patients": self.patients,
            "positives": self.positives,
            "prevalence": self.prevalence,
            "feature_columns": self.feature_columns,
            "imputed_fraction": self.imputed_fraction,
            "output": self.output,
            "feature_scaling": "clinical units; standardisation fitted on each training split in data.py",
        }


def load_itemid_map(path: Path | str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path or RESOURCE_MAP)
    if set(frame.columns) < {"variable", "itemid", "linksto"}:
        raise ValueError("itemid map needs variable, itemid and linksto columns")
    return frame


def variable_order(itemid_map: pd.DataFrame) -> list[str]:
    """The 17 channels in a fixed order.

    Sorted rather than taken in file order so that two runs on different
    machines produce byte-identical column names, which is what lets a cohort
    built today be compared against one built after a re-download.
    """
    return sorted(itemid_map["variable"].unique())


def _read_table(mimic_dir: Path, name: str, **kwargs) -> pd.DataFrame:
    """Read a MIMIC table, accepting either the plain or gzipped distribution."""
    for candidate in (mimic_dir / f"{name}.csv", mimic_dir / f"{name}.csv.gz"):
        if candidate.exists():
            frame = pd.read_csv(candidate, **kwargs)
            if isinstance(frame, pd.DataFrame):
                frame.columns = frame.columns.str.lower()
            return frame
    raise FileNotFoundError(f"{name}.csv not found under {mimic_dir}")


def _iter_events(mimic_dir: Path, name: str, usecols: list[str], chunksize: int):
    """Stream an events table.

    CHARTEVENTS is ~33 GB in the full distribution, so it is filtered chunk by
    chunk against the cohort rather than loaded. The demo distribution is small
    enough that this costs nothing there.
    """
    for candidate in (mimic_dir / f"{name}.csv", mimic_dir / f"{name}.csv.gz"):
        if candidate.exists():
            reader = pd.read_csv(
                candidate,
                usecols=lambda c: c.lower() in usecols,
                chunksize=chunksize,
                low_memory=False,
            )
            for chunk in reader:
                chunk.columns = chunk.columns.str.lower()
                yield chunk
            return
    raise FileNotFoundError(f"{name}.csv not found under {mimic_dir}")


def select_cohort(mimic_dir: Path, min_los_days: float = 2.0, min_age: int = 18) -> pd.DataFrame:
    """Adult ICU stays long enough to carry a 24-hour window and a 24-hour gap."""
    icu = _read_table(mimic_dir, "ICUSTAYS")
    patients = _read_table(mimic_dir, "PATIENTS")
    admissions = _read_table(mimic_dir, "ADMISSIONS")

    cohort = icu.merge(patients[["subject_id", "dob"]], on="subject_id", how="inner").merge(
        admissions[["hadm_id", "hospital_expire_flag", "deathtime"]], on="hadm_id", how="inner"
    )
    cohort["intime"] = pd.to_datetime(cohort["intime"])

    # Dates of birth above 89 are shifted back ~300 years, which overflows a
    # timedelta. Calendar-year differencing avoids it; shifted ages fold to 90.
    born = pd.to_datetime(cohort["dob"], errors="coerce")
    cohort["age"] = cohort["intime"].dt.year - born.dt.year
    cohort.loc[cohort["age"] > 89, "age"] = 90

    cohort = cohort[(cohort["age"] >= min_age) & (cohort["los"] > min_los_days)]
    # First stay per admission, so one deterioration yields one labelled window.
    cohort = cohort.sort_values("intime").groupby("hadm_id", as_index=False).first()
    return cohort[["icustay_id", "hadm_id", "subject_id", "intime", "hospital_expire_flag", "age"]]


def _harmonise_units(events: pd.DataFrame) -> pd.DataFrame:
    """Put every channel on one unit, then drop what is out of range.

    MIMIC records temperature in both Fahrenheit and Celsius, oxygen fraction as
    both a percentage and a fraction, and weight in both pounds and kilograms.
    The declared unit wins wherever there is one; magnitude only decides the
    readings that were charted without it.
    """
    value = events["valuenum"].astype(float)
    variable = events["variable"]
    if "valueuom" in events.columns:
        uom = events["valueuom"].fillna("").astype(str).str.strip().str.lower()
    else:
        uom = pd.Series("", index=events.index, dtype=object)
    blank = uom == ""

    # MetaVision writes "Deg. F"/"Deg. C"; the CareVue export carries a mojibake
    # degree sign, "?F"/"?C". No Celsius spelling contains an "f".
    is_temp = variable == "Temperature"
    fahrenheit = is_temp & (uom.str.contains("f", regex=False) | (blank & (value > 70.0)))
    value = value.where(~fahrenheit, (value - 32.0) * 5.0 / 9.0)

    # A fraction charted in torr is a partial pressure, so it is dropped rather
    # than rescaled into the valid 0.2-1.0 band.
    is_fio2 = variable == "Fraction inspired oxygen"
    torr = is_fio2 & uom.str.contains("torr", regex=False)
    percent = is_fio2 & ~torr & (value > 1.0)
    value = value.where(~percent, value / 100.0)

    is_weight = variable == "Weight"
    if "itemid" in events.columns:
        itemid = pd.to_numeric(events["itemid"], errors="coerce")
    else:
        itemid = pd.Series(np.nan, index=events.index, dtype=float)
    ounces = is_weight & (uom.isin(OUNCE_UNITS) | (blank & itemid.isin(OUNCE_WEIGHT_ITEMIDS)))
    value = value.where(~ounces, value / 16.0)
    pounds = (
        (is_weight & uom.isin(POUND_UNITS))
        | (is_weight & blank & itemid.isin(POUND_WEIGHT_ITEMIDS))
        | ounces
        | (is_weight & blank & itemid.isna() & (value > 300.0))
    )
    value = value.where(~pounds, value * 0.45359237)

    is_height = variable == "Height"
    inches = (is_height & uom.isin(INCH_UNITS)) | (is_height & blank & (value < 100.0))
    value = value.where(~inches, value * 2.54)

    events = events.assign(valuenum=value).loc[~torr]
    low = events["variable"].map(lambda v: VALID_RANGES[v][0])
    high = events["variable"].map(lambda v: VALID_RANGES[v][1])
    return events[(events["valuenum"] >= low) & (events["valuenum"] <= high)]


def _score_text_values(events: pd.DataFrame) -> pd.DataFrame:
    """Recover a number for the channels charted as a label, not a value.

    Four of the seventeen channels are ordinal or binary and are written as
    words: the three GCS sub-scores and capillary refill. Their ``valuenum`` is
    empty, so a reading that was in fact taken is indistinguishable from one
    that was never taken, and the channel reads as never observed for the whole
    cohort. Scoring the label first is what keeps that distinction.
    """
    missing = events["valuenum"].isna()
    if not missing.any():
        return events
    for variable, table in TEXT_TO_SCORE.items():
        rows = missing & (events["variable"] == variable)
        if not rows.any():
            continue
        text = events.loc[rows, "value"].astype(str).str.strip().str.lower()
        events.loc[rows, "valuenum"] = text.map(table)
    return events


def collect_events(
    mimic_dir: Path, cohort: pd.DataFrame, itemid_map: pd.DataFrame, chunksize: int = 2_000_000
) -> pd.DataFrame:
    """Every in-window observation of the 17 channels, one row per reading."""
    by_table: dict[str, pd.DataFrame] = {
        table: group for table, group in itemid_map.groupby("linksto")
    }
    stay_start = cohort.set_index("icustay_id")["intime"]
    wanted_stays = set(cohort["icustay_id"])
    hadm_start = cohort.set_index("hadm_id")["intime"]
    wanted_hadm = set(cohort["hadm_id"])

    collected: list[pd.DataFrame] = []
    var_dtype = pd.CategoricalDtype(sorted(itemid_map["variable"].unique()))
    for table, group in by_table.items():
        name = "CHARTEVENTS" if table == "chartevents" else "LABEVENTS"
        lookup = dict(zip(group["itemid"], group["variable"]))
        # LABEVENTS carries no icustay_id, so lab readings use the admission's
        # ICU clock; the filter above leaves one stay per admission.
        keyed_by_stay = name == "CHARTEVENTS"
        cols = ["itemid", "charttime", "valuenum", "value", "valueuom"]
        cols += ["icustay_id"] if keyed_by_stay else ["hadm_id"]

        for chunk in _iter_events(mimic_dir, name, cols, chunksize):
            chunk = chunk[chunk["itemid"].isin(lookup)]
            if chunk.empty:
                continue
            key = "icustay_id" if keyed_by_stay else "hadm_id"
            wanted = wanted_stays if keyed_by_stay else wanted_hadm
            chunk = chunk[chunk[key].isin(wanted)]
            if chunk.empty:
                continue
            chunk = chunk.assign(variable=chunk["itemid"].map(lookup))
            chunk = _score_text_values(chunk)
            chunk = chunk[chunk["valuenum"].notna()]
            if chunk.empty:
                continue
            chunk = _harmonise_units(chunk)
            if chunk.empty:
                continue
            origin = (stay_start if keyed_by_stay else hadm_start).reindex(chunk[key]).to_numpy()
            hour = (pd.to_datetime(chunk["charttime"]).to_numpy() - origin) / np.timedelta64(1, "h")
            chunk = chunk.assign(hour=np.floor(hour))
            if not keyed_by_stay:
                stay_of_hadm = cohort.set_index("hadm_id")["icustay_id"]
                chunk = chunk.assign(icustay_id=stay_of_hadm.reindex(chunk["hadm_id"]).to_numpy())
            chunk = chunk[(chunk["hour"] >= 0) & (chunk["hour"] < TIMESTEPS)]
            if not chunk.empty:
                kept = chunk[["icustay_id", "variable", "hour", "valuenum"]]
                # Categorical storage: the channel name repeats across tens of
                # millions of retained rows in the full distribution.
                kept = kept.astype({"variable": var_dtype})
                collected.append(kept)

    if not collected:
        return pd.DataFrame(columns=["icustay_id", "variable", "hour", "valuenum"])
    return pd.concat(collected, ignore_index=True)


def build_windows(
    cohort: pd.DataFrame, events: pd.DataFrame, variables: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """A dense (stays, 24, 17) tensor, plus the fraction of cells imputed.

    Several readings can land in the same hour, so they are reduced by the
    median: a single mistyped extreme then cannot move the hour, which a mean
    would let it do.
    """
    stays = cohort["icustay_id"].to_numpy()
    index = {stay: i for i, stay in enumerate(stays)}
    channel = {name: j for j, name in enumerate(variables)}

    grid = np.full((len(stays), TIMESTEPS, len(variables)), np.nan, dtype=np.float64)
    if not events.empty:
        hourly = events.groupby(
            ["icustay_id", "hour", "variable"], as_index=False, observed=True
        )["valuenum"].median()
        rows = hourly["icustay_id"].map(index).to_numpy()
        hours = hourly["hour"].astype(int).to_numpy()
        cols = hourly["variable"].map(channel).to_numpy()
        keep = ~pd.isna(rows)
        grid[rows[keep].astype(int), hours[keep], cols[keep].astype(int)] = (
            hourly["valuenum"].to_numpy()[keep]
        )

    observed = np.isfinite(grid)
    # Forward-fill along time within each stay, then fall back to NORMAL_VALUES.
    for t in range(1, TIMESTEPS):
        missing = ~np.isfinite(grid[:, t, :])
        grid[:, t, :][missing] = grid[:, t - 1, :][missing]
    for name, j in channel.items():
        column = grid[:, :, j]
        column[~np.isfinite(column)] = NORMAL_VALUES[name]
        grid[:, :, j] = column

    return grid, observed


def zscore(grid: np.ndarray, training_reference: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Transform with explicitly supplied training rows, never an unsplit cohort."""
    mean = training_reference.mean(axis=(0, 1))
    std = training_reference.std(axis=(0, 1))
    std = np.where(std < 1e-8, 1.0, std)
    return (grid - mean) / std, {"mean": mean.tolist(), "std": std.tolist()}


def build_cohort_csv(
    mimic_dir: Path | str,
    output: Path | str = "data/processed/mimic_mortality_24h.csv",
    chunksize: int = 2_000_000,
) -> CohortReport:
    """Run the whole pipeline and write the flattened cohort."""
    mimic_dir = Path(mimic_dir)
    output = Path(output)
    itemid_map = load_itemid_map()
    variables = variable_order(itemid_map)

    cohort = select_cohort(mimic_dir)
    if cohort.empty:
        raise ValueError("no ICU stay survived the adult / >48 h filters")
    events = collect_events(mimic_dir, cohort, itemid_map, chunksize)
    grid, observed = build_windows(cohort, events, variables)

    # Timestep-major, matching the reshape in data.py.
    columns = [f"t{t:02d}_{name}" for t in range(TIMESTEPS) for name in variables]
    # Preserve clinical units here. Each experiment seed defines its own patient
    # split, so normalisation is fitted later using only that split's train rows.
    flat = grid.reshape(len(cohort), TIMESTEPS * len(variables))
    frame = pd.DataFrame(flat, columns=columns)
    frame.insert(0, "subject_id", cohort["subject_id"].to_numpy())
    # Legacy column name retained for existing configs; this is in-hospital
    # mortality, not death within 48 hours of the observation window.
    frame["mortality_48h"] = cohort["hospital_expire_flag"].to_numpy().astype(int)

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    positives = int(frame["mortality_48h"].sum())
    return CohortReport(
        stays=len(frame),
        patients=int(frame["subject_id"].nunique()),
        positives=positives,
        prevalence=positives / len(frame),
        feature_columns=len(columns),
        imputed_fraction=float(1.0 - observed.mean()),
        output=str(output),
    )


if __name__ == "__main__":  # pragma: no cover
    report = build_cohort_csv(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else
                              "data/processed/mimic_mortality_24h.csv")
    for key, value in report.to_dict().items():
        print(f"{key}={value}")
