#!/usr/bin/env python3
"""Prepare the nine public Dataset 3–5 inputs for scale-aware sensitivity.

Use existing canonical raw/ inputs or prepare a local working copy of the
manuscript's Supplementary Data 1 as described in README.md. Then run:
    python3 prepare_revision3_inputs.py --check
    python3 prepare_revision3_inputs.py

The original raw/ workbooks remain unchanged. Eight matrices are copied to
raw_revision3/ without changing their cells. For Dataset 5 Post2, the supplied
source-derived mask restores only the 3,275 documented missing cells, while
preserving 24 genuine observed zeros. No model is fitted by this script.
"""
from __future__ import annotations

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "raw"
OUTPUT_DIR = SCRIPT_DIR / "raw_revision3"
SUPPORT_DIR = SCRIPT_DIR / "revision3_inputs"
DATASETS = tuple(f"dataset{d}_{t}" for d in (3, 4, 5) for t in ("pre", "post1", "post2"))

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import numbers
import posixpath
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import openpyxl
from openpyxl.utils import get_column_letter

MISSING_TOKENS = {"", "NA", "N/A", "na", "n/a"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[list[object]]:
    book = openpyxl.load_workbook(path, read_only=True, data_only=False)
    try:
        if book.sheetnames != ["Sheet1"]:
            raise ValueError(f"Expected one Sheet1 worksheet: {path}")
        rows = [list(row) for row in book["Sheet1"].values if any(v is not None for v in row)]
    finally:
        book.close()
    if len(rows) < 2 or rows[0][:2] not in (["id", "category"], ["id", "Responder"]):
        raise ValueError(f"Expected id/category (or Responder) headers: {path}")
    if len(set(rows[0])) != len(rows[0]) or any(not isinstance(v, str) or not v for v in rows[0]):
        raise ValueError(f"Malformed or duplicate headers: {path}")
    ids = [str(row[0]).strip() for row in rows[1:]]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"Missing or duplicate IDs: {path}")
    for row in rows[1:]:
        if isinstance(row[1], bool) or row[1] not in (0, 1):
            raise ValueError(f"Expected binary numeric outcome: {path}")
        for value in row[2:]:
            if value is None or (isinstance(value, str) and value in MISSING_TOKENS):
                continue
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
                raise ValueError(f"Non-numeric or non-finite feature: {path}")
    return rows


def semantic_sha256(rows: list[list[object]]) -> str:
    header = list(rows[0])
    header[1] = "category"
    normalized = [header]
    for row in rows[1:]:
        features = ["missing" if v is None or (isinstance(v, str) and v in MISSING_TOKENS)
                    else float(v).hex() for v in row[2:]]
        normalized.append([str(row[0]).strip(), str(int(row[1])), *features])
    text = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sheet_member(archive: zipfile.ZipFile) -> str:
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheet = next(v for v in workbook.findall("s:sheets/s:sheet", ns) if v.get("name") == "Sheet1")
    rel = sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    target = next(v.get("Target") for v in relationships if v.get("Id") == rel)
    return target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join("xl", target))


def restore_mask(current: Path, output: Path, mask: list[dict[str, str]]) -> None:
    book = openpyxl.load_workbook(current, read_only=True, data_only=False)
    try:
        rows = list(book["Sheet1"].values)
    finally:
        book.close()
    header = list(rows[0])
    positions = {str(row[0]).strip(): r for r, row in enumerate(rows[1:], start=2) if any(v is not None for v in row)}
    columns = {name: c for c, name in enumerate(header, start=1)}
    wanted = set()
    for item in mask:
        row, col = positions[item["id"]], columns[item["feature"]]
        if rows[row - 1][col - 1] != 0:
            raise ValueError(f"Restoration cell is not zero: {item['id']}/{item['feature']}")
        wanted.add(f"{get_column_letter(col)}{row}".encode("ascii"))
    if len(wanted) != 3275 or len(mask) != 3275:
        raise ValueError("Missingness mask must contain 3,275 unique cells")
    with zipfile.ZipFile(current) as before:
        member = sheet_member(before)
        xml = before.read(member)
        seen = set()

        def replace(match):
            coordinate = match.group(2)
            if coordinate not in wanted:
                return match.group(0)
            if coordinate in seen:
                raise ValueError("Duplicate XLSX cell")
            seen.add(coordinate)
            body = match.group(3)
            values = list(re.finditer(rb"<v(?:\s[^>]*)?>([^<]*)</v>", body))
            if re.search(rb"<f(?:\s|>)", body) or len(values) != 1 or float(values[0].group(1)) != 0:
                raise ValueError("Expected an ordinary zero-valued XLSX cell")
            value = values[0]
            return match.group(1) + body[:value.start()] + body[value.end():] + match.group(4)

        xml = re.sub(rb'(<c\b[^>]*\br="([^"]+)"[^>]*>)(.*?)(</c>)', replace, xml, flags=re.DOTALL)
        if seen != wanted:
            raise ValueError("Some missingness-mask cells were not located")
        with zipfile.ZipFile(output, "x") as after:
            for entry in before.infolist():
                after.writestr(entry, xml if entry.filename == member else before.read(entry.filename))
    old_rows, new_rows = read_rows(current), read_rows(output)
    changed = {(item["id"], item["feature"]) for item in mask}
    if old_rows[0] != new_rows[0] or len(old_rows) != len(new_rows):
        raise ValueError("Restoration changed row/column layout")
    zeros = 0
    for old, new in zip(old_rows[1:], new_rows[1:]):
        if old[:2] != new[:2]:
            raise ValueError("Restoration changed patient identity or outcome")
        for name, a, b in zip(old_rows[0][2:], old[2:], new[2:]):
            if (str(old[0]).strip(), name) in changed:
                if a != 0 or b is not None:
                    raise ValueError("Incorrect missingness restoration")
            elif a != b:
                raise ValueError("Restoration changed an observed value")
            else:
                zeros += int(b == 0)
    if zeros != 24:
        raise ValueError(f"Expected 24 genuine observed zeros; found {zeros}")


def prepare(input_dir: Path, output_dir: Path, *, check: bool = False) -> dict:
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()
    if input_dir == output_dir:
        raise ValueError("Original and sensitivity input folders must differ")
    manifest = json.loads((SUPPORT_DIR / "input_manifest.json").read_text(encoding="utf-8"))
    mask_path = SUPPORT_DIR / manifest["mask_file"]
    if sha256(mask_path) != manifest["mask_sha256"]:
        raise ValueError("The supplied missingness mask has changed")
    with mask_path.open(encoding="utf-8", newline="") as handle:
        mask = list(csv.DictReader(handle))
    original = {}
    for key in DATASETS:
        path = input_dir / f"{key}.xlsx"
        rows = read_rows(path)
        expected = manifest["datasets"][key]
        counts = {"samples": len(rows)-1, "features": len(rows[0])-2,
                  "positive": sum(int(row[1]) == 1 for row in rows[1:]),
                  "negative": sum(int(row[1]) == 0 for row in rows[1:])}
        if any(counts[name] != expected[name] for name in counts):
            raise ValueError(f"Unexpected dimensions or outcome counts: {key}")
        # Accept only the two verified representations: the original workbook
        # and prepare_inputs.py's XLSX export (18 cells differ by <= 8.89e-16).
        accepted = {expected["semantic_sha256"], expected["canonical_export_semantic_sha256"]}
        if semantic_sha256(rows) not in accepted:
            raise ValueError(f"Input values/order differ from the supplied main-analysis matrix: {key}")
        original[key] = {"sha256": sha256(path), "semantic_sha256": semantic_sha256(rows), **counts}
    if check:
        return {"status": "validated", "analysis_run": False, "datasets": list(DATASETS),
                "input_files": original, "repair_counts": manifest["repair_counts"]}
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; keep it or choose a new --out folder: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare_revision3_", dir=output_dir.parent) as temp:
        stage = Path(temp) / "inputs"
        stage.mkdir()
        produced = {}
        for key in DATASETS:
            source, destination = input_dir / f"{key}.xlsx", stage / f"{key}.xlsx"
            if key == "dataset5_post2":
                restore_mask(source, destination, mask)
            else:
                shutil.copyfile(source, destination)
            rows = read_rows(destination)
            produced[key] = {"path": f"{key}.xlsx", "sha256": sha256(destination),
                             "semantic_sha256": semantic_sha256(rows),
                             **{n: original[key][n] for n in ("samples", "features", "positive", "negative")}}
        for key, item in original.items():
            if sha256(input_dir / f"{key}.xlsx") != item["sha256"]:
                raise ValueError("An original input changed during preparation")
        receipt = {"status": "completed", "analysis_run": False,
                   "preprocessing": "scale_aware", "created_utc": datetime.now(timezone.utc).isoformat(),
                   "datasets": list(DATASETS), "inputs": produced,
                   "source_inputs": original, "repair_counts": manifest["repair_counts"],
                   "mask_sha256": manifest["mask_sha256"], "script_sha256": sha256(Path(__file__)),
                   "note": "Eight matrices copied; only documented source-missing D5 Post2 zeros restored. No log transformation or model fitting performed."}
        (stage / "input_preparation_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        stage.rename(output_dir)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--check", action="store_true", help="Validate the original inputs and restoration mask without creating outputs")
    args = parser.parse_args()
    receipt = prepare(args.input_dir, args.out, check=args.check)
    print(f"{receipt['status']}: nine Dataset 3–5 matrices; 3,275 source-missing D5 Post2 cells identified; no model fitting.")
    if not args.check:
        print(f"Prepared inputs: {args.out.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
