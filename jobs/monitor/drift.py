"""Feature and text drift with Evidently (PLAN.md, section 5, trigger 2).

Two reports, because Evidently decides text drift with a domain classifier whose cutoff isn't in
its output (F-121): metadata features in one report, giving the share that drifted, and the added
text in the other, where any drift counts. Each column is tested on its own, so splitting them
changes no result.
"""

from pathlib import Path
from typing import Any

import pandas as pd

METADATA_COLUMNS = (
    "bytes_delta",
    "chars_added",
    "uppercase_ratio",
    "badword_count",
    "hour_of_day",
    "page_length",
    "user_edit_count_bucket",
)
CATEGORICAL_COLUMNS = ("is_anonymous", "is_minor", "has_external_link")
TEXT_COLUMNS = ("added_text",)


def _frame(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.DataFrame([{name: row[name] for name in columns} for row in rows])
    for name in CATEGORICAL_COLUMNS:
        if name in frame:
            frame[name] = frame[name].astype(int)
    return frame


def _drift_columns(report: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], float, int]:
    """Per column: the statistic, the test Evidently chose, and its threshold. Then the share and
    count it says drifted.

    The test depends on the column and the sample size: past 1,000 rows, numbers use Wasserstein
    distance and categories Jensen-Shannon, where higher means more drift; smaller samples use
    p-values, where lower does (F-124). So a value means nothing without its method.
    """
    values: dict[str, dict[str, Any]] = {}
    share, count = 0.0, 0
    for metric in report["metrics"]:
        config = metric.get("config", {})
        if config.get("type", "").endswith("DriftedColumnsCount"):
            share = float(metric["value"]["share"])
            count = int(metric["value"]["count"])
        elif config.get("type", "").endswith("ValueDrift"):
            values[config["column"]] = {
                "value": float(metric["value"]),
                "method": str(config.get("method", "")),
                "threshold": config.get("threshold"),
            }
    return values, share, count


def evidently_drift(
    reference: list[dict[str, Any]], current: list[dict[str, Any]], html_folder: Path | None = None
) -> dict[str, Any]:
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    # A column that never varies in the reference can't be compared: backfilled rows have no edit
    # count (DEC-036), so live counts would always look like drift (F-122).
    constant = [
        name
        for name in (*METADATA_COLUMNS, *CATEGORICAL_COLUMNS)
        if len({row[name] for row in reference}) <= 1
    ]
    numerical = [name for name in METADATA_COLUMNS if name not in constant]
    categorical = [name for name in CATEGORICAL_COLUMNS if name not in constant]
    meta_columns = (*numerical, *categorical)
    meta = DataDefinition(numerical_columns=numerical, categorical_columns=categorical)
    meta_run = Report([DataDriftPreset()]).run(
        Dataset.from_pandas(_frame(current, meta_columns), data_definition=meta),
        Dataset.from_pandas(_frame(reference, meta_columns), data_definition=meta),
    )
    text = DataDefinition(text_columns=list(TEXT_COLUMNS))
    text_run = Report([DataDriftPreset()]).run(
        Dataset.from_pandas(_frame(current, TEXT_COLUMNS), data_definition=text),
        Dataset.from_pandas(_frame(reference, TEXT_COLUMNS), data_definition=text),
    )
    statistics, share, count = _drift_columns(meta_run.dict())
    text_values, _, text_count = _drift_columns(text_run.dict())
    if html_folder is not None:
        html_folder.mkdir(parents=True, exist_ok=True)
        meta_run.save_html(str(html_folder / "drift_metadata.html"))
        text_run.save_html(str(html_folder / "drift_text.html"))
    return {
        "method": "evidently DataDriftPreset",
        "metadata_share_drifted": share,
        "metadata_drifted_count": count,
        "metadata_columns": len(meta_columns),
        "metadata_statistics": statistics,
        "not_comparable": constant,
        "text_drift": text_count > 0,
        "text_statistics": text_values,
        "reference_rows": len(reference),
        "current_rows": len(current),
    }
