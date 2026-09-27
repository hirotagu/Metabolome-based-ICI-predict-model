"""Known study inputs and strict, subset-aware dataset selection.

Availability records file presence only; it does not establish sharing permission.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Union

# Keep the original Supplementary Data 1 mapping; absent sheets are never renumbered.
SHEET_DATASET_MAP = (
    ("Sheet1", "dataset1_pre"), ("Sheet2", "dataset2_pre"),
    ("Sheet3", "dataset3_pre"), ("Sheet4", "dataset4_pre"),
    ("Sheet5", "dataset5_pre"), ("Sheet6", "dataset6_pre"),
    ("Sheet7", "dataset7_pre"), ("Sheet8", "dataset1_post1"),
    ("Sheet9", "dataset2_post1"), ("Sheet10", "dataset3_post1"),
    ("Sheet11", "dataset4_post1"), ("Sheet12", "dataset5_post1"),
    ("Sheet13", "dataset6_post1"), ("Sheet14", "dataset2_post2"),
    ("Sheet15", "dataset3_post2"), ("Sheet16", "dataset4_post2"),
    ("Sheet17", "dataset5_post2"),
)
KNOWN_DATASET_KEYS = tuple(key for _, key in SHEET_DATASET_MAP)
FORMAL_PRE_KEYS = tuple(f"dataset{number}_pre" for number in (1, 2, 3, 4, 5, 7))


def canonical_key(value: str) -> str:
    key = str(value).strip().casefold()
    if key not in KNOWN_DATASET_KEYS:
        raise ValueError(f"Unknown study dataset: {value!r}; choose from {KNOWN_DATASET_KEYS}")
    return key


def _unique_keys(values: Iterable[str], name: str) -> List[str]:
    keys = [canonical_key(value) for value in values]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"Duplicate {name} dataset(s): {duplicates}")
    return keys


def select_available(
    selector: Union[str, Sequence[str]],
    available: Iterable[str],
    default_keys: Optional[Iterable[str]] = None,
) -> List[str]:
    """Intersect ALL with a profile, but fail for explicit missing requests.

    ALL_PRIMARY is retained as a compatibility synonym for ALL.  ``available``
    must already be a unique inventory of recognized input files.
    """
    present = _unique_keys(available, "available")
    present_set = set(present)
    if isinstance(selector, str) and selector.strip().upper() in {"ALL", "ALL_PRIMARY"}:
        defaults = present if default_keys is None else _unique_keys(default_keys, "default")
        return [key for key in defaults if key in present_set]
    requested = _unique_keys([selector] if isinstance(selector, str) else selector, "requested")
    missing = [key for key in requested if key not in present_set]
    if missing:
        raise FileNotFoundError(f"Explicitly requested dataset input(s) not provided: {missing}")
    return requested


def coverage_rows(
    available: Iterable[str],
    selected: Iterable[str] = (),
    expected: Iterable[str] = KNOWN_DATASET_KEYS,
) -> List[dict]:
    """Return explicit coverage, without inferring authorization from presence."""
    present = set(_unique_keys(available, "available"))
    chosen = set(_unique_keys(selected, "selected"))
    if not chosen <= present:
        raise ValueError(f"Selected dataset(s) not present: {sorted(chosen - present)}")
    return [
        {
            "dataset": key,
            "input_status": "available" if key in present else "not_provided",
            "selection_status": "selected" if key in chosen else "not_selected",
        }
        for key in _unique_keys(expected, "expected")
    ]
