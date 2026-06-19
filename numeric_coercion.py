from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd


def _strip_if_str(value):
    return value.strip() if isinstance(value, str) else value


def normalize_numeric_like(obj: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """Normalize common human-formatted numeric strings.

    - Strips surrounding whitespace
    - Treats empty/whitespace-only strings as missing (pd.NA)
    - Removes thousand separators (",")
    - Removes percent signs ("%")

    Notes
    -----
    - Percent signs are *removed* but values are NOT divided by 100.
      E.g. "12%" -> "12" -> 12.0
    """

    if isinstance(obj, pd.DataFrame):
        cleaned = obj.apply(lambda col: col.map(_strip_if_str))
        cleaned = cleaned.replace(r"^\s*$", pd.NA, regex=True)
        cleaned = cleaned.replace({",": "", "%": ""}, regex=True)
        return cleaned

    if isinstance(obj, pd.Series):
        cleaned = obj.map(_strip_if_str)
        cleaned = cleaned.replace(r"^\s*$", pd.NA, regex=True)
        cleaned = cleaned.replace({",": "", "%": ""}, regex=True)
        return cleaned

    raise TypeError(f"Unsupported type: {type(obj)!r}")


def coerce_numeric_series(series: pd.Series, *, errors: str = "coerce") -> pd.Series:
    """Coerce a series to numeric after normalizing numeric-like strings."""

    cleaned = normalize_numeric_like(series)
    return pd.to_numeric(cleaned, errors=errors)


# ---------------------------------------------------------------------------
# Shared helpers for detecting header rows and classifying cell types.
# These were previously duplicated in box_plot / piechart_plot / trend_plot /
# pareto_plot.  Keep a single definition here so every renderer uses the same
# logic. -------------------------------------------------------------------


def is_blank_cell(value) -> bool:
    """Return True when *value* represents an empty / missing cell."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def is_numeric_type_cell(value) -> bool:
    """Return True when *value* is already a Python/NumPy numeric type.

    Strings like ``"1"`` / ``"1.2"`` are *not* numeric here – they should
    trigger header detection (because a header row is "any cell that is
    non-empty and not a raw number").

    ``bool`` is explicitly excluded because ``True`` / ``False`` show up in
    Excel cells but ``bool`` is a subclass of ``int``.
    """
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float, np.number))


def detect_header_row(df: pd.DataFrame) -> bool:
    """Auto-detect whether the first row of *df* is a header row.

    The heuristic: if any non-blank cell in the first row is *not* a raw
    numeric type, the row is treated as a header.
    """
    if df is None or df.shape[0] == 0 or df.shape[1] == 0:
        return False

    first_row = df.iloc[0, :].tolist()
    for cell in first_row:
        if is_blank_cell(cell):
            continue
        if not is_numeric_type_cell(cell):
            return True
    return False
