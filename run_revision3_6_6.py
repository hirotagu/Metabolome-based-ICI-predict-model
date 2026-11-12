#!/usr/bin/env python3
"""Reproduce the scale-aware sensitivity analyses for Datasets 3–5.

Prepare inputs with prepare_revision3_inputs.py first, then run:
    python3 run_revision3_6_6.py --check  # input checks only; no model fitting
    python3 run_revision3_6_6.py          # all analyses; resumes completed jobs

The main-analysis code remains separate. No private results archive is needed.
"""
from __future__ import annotations
from pathlib import Path

# USER SETTINGS
PACKAGE_DIR = Path(__file__).resolve().parent
INPUT_DIR = PACKAGE_DIR / "raw_revision3"
OUTPUT_DIR = PACKAGE_DIR / "result_revision3"
PLOT_FILES = {f"FigureS_scale_sensitivity.{suffix}" for suffix in ("pdf", "png", "csv", "json")}

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import signal
import subprocess
import sys

ENGINE = PACKAGE_DIR / "revision3_engine"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ENGINE))
KEYS = tuple(f"dataset{i}_{t}" for i in (3, 4, 5)
             for t in ("pre", "post1", "post2"))
EXPECTED = {
    "dataset3_pre": (77, 57, 20, 202), "dataset3_post1": (71, 55, 16, 202),
    "dataset3_post2": (61, 51, 10, 202), "dataset4_pre": (72, 49, 23, 107),
    "dataset4_post1": (64, 46, 18, 107), "dataset4_post2": (61, 47, 14, 107),
    "dataset5_pre": (381, 199, 182, 202), "dataset5_post1": (95, 60, 35, 202),
    "dataset5_post2": (314, 183, 131, 202),
}
VERSIONS = {"numpy": "2.3.3", "pandas": "2.3.3", "scikit-learn": "1.7.2",
            "scipy": "1.16.2", "openpyxl": "3.1.5"}
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
for _name in THREAD_VARS:
    os.environ[_name] = "1"
os.environ.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def native(value):
    """Normalize receipts only; numerical analysis files are not rounded."""
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return native(value.tolist())
    if hasattr(value, "item"):
        return native(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(native(value), indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def runtime(strict):
    versions, differences = {}, []
    for name, expected in VERSIONS.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Missing {name}; install requirements.txt in this Python environment.") from exc
        versions[name] = actual
        if actual != expected:
            differences.append(f"{name}={actual} (expected {expected})")
    if sys.version_info[:2] != (3, 11):
        differences.insert(0, f"Python {platform.python_version()} (use Python 3.11; original run: 3.11.16)")
    if differences:
        message = "Environment differs from the manuscript run: " + "; ".join(differences)
        if strict:
            raise RuntimeError(message)
        print("WARNING: " + message, flush=True)
    return {"python": platform.python_version(), "executable": str(Path(sys.executable).resolve()),
            "system": platform.system(), "machine": platform.machine(), "packages": versions,
            "thread_limits": {k: "1" for k in THREAD_VARS}}


def code_identity():
    files = [Path(__file__).resolve(), *sorted(ENGINE.glob("*.py"))]
    if not (ENGINE / "ICI_predict_v3.py").is_file():
        raise FileNotFoundError("revision3_engine/ is missing; extract the complete public package.")
    return {p.relative_to(PACKAGE_DIR).as_posix(): sha(p) for p in files}


def build_plan():
    import core_jobs
    import sensitivity_jobs
    jobs = core_jobs.plan_jobs() + sensitivity_jobs.plan_jobs()
    ordered, pending = [], list(jobs)
    while pending:
        done = {j["id"] for j in ordered}
        candidates = [j for j in pending if set(j.get("depends_on", [])).issubset(done)]
        if not candidates:
            raise ValueError("Analysis plan contains an unknown dependency or a cycle.")
        job = candidates[0]
        if "dataset" in job and job["dataset"] not in KEYS:
            raise ValueError(f"Out-of-scope fit: {job['id']}")
        ordered.append(job)
        pending.remove(job)
    if len({j["id"] for j in ordered}) != len(ordered):
        raise ValueError("Duplicate job identifier.")
    claims = []
    for job in ordered:
        for rel in job["owned_paths"]:
            path = Path(rel)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Invalid output path: {rel}")
            if any(path == old or path in old.parents or old in path.parents for old in claims):
                raise ValueError(f"Overlapping output ownership: {rel}")
            claims.append(path)
    return ordered


def validate_inputs(input_dir):
    import ICI_predict_v3 as base
    import numpy as np
    import openpyxl
    input_dir = Path(input_dir).expanduser().resolve()
    inputs = {key: input_dir / f"{key}.xlsx" for key in KEYS}
    missing = [str(p) for p in inputs.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Prepare inputs with prepare_revision3_inputs.py. Missing: " + ", ".join(missing))
    preparation_path = input_dir / "input_preparation_receipt.json"
    if not preparation_path.is_file():
        raise FileNotFoundError("Missing input_preparation_receipt.json. Prepare these inputs "
                                "with prepare_revision3_inputs.py rather than copying the old matrices.")
    preparation = read_json(preparation_path)
    if (preparation.get("status") != "completed" or
            preparation.get("preprocessing") != "scale_aware" or
            set(preparation.get("datasets", [])) != set(KEYS) or
            set(preparation.get("inputs", {})) != set(KEYS)):
        raise ValueError("Invalid scale-aware input preparation receipt.")
    for key, file in inputs.items():
        if preparation["inputs"][key].get("sha256") != sha(file):
            raise ValueError(f"Prepared input has changed: {key}. Prepare a new input folder.")
    base.INPUT_DIR = input_dir
    base.INPUT_FILES = [str(inputs[k]) for k in KEYS]
    base.DATASETS_TO_RUN = list(KEYS)
    base.PRIMARY_TARGETS = list(KEYS)
    specs = base.discover_datasets()
    if set(specs) != set(KEYS):
        raise ValueError("Expected the nine Dataset 3–5 matrices.")
    checks = []
    for key in KEYS:
        bundle = base.load_dataset(specs[key], specs[key].primary_sheet)
        actual = (len(bundle.y), int(np.sum(bundle.y == 1)), int(np.sum(bundle.y == 0)),
                  len(bundle.feature_manifest))
        if actual != EXPECTED[key]:
            raise ValueError(f"{key}: dimensions/classes/features {actual}; expected {EXPECTED[key]}")
        checks.append({"dataset": key, "n": actual[0], "positive": actual[1],
                       "negative": actual[2], "input_features": actual[3],
                       "available_candidates": len(bundle.feature_names), "input_scale": "log10",
                       "path": str(inputs[key]), "sha256": sha(inputs[key])})
    workbook = openpyxl.load_workbook(inputs["dataset5_post2"], read_only=True, data_only=False)
    try:
        rows = list(workbook["Sheet1"].values)
        data = [v for row in rows[1:] for v in row[2:204]]
        blanks = sum(v is None for v in data)
        zeros = sum(isinstance(v, (float, int)) and not isinstance(v, bool) and v == 0 for v in data)
        if (blanks, zeros) != (3275, 24):
            raise ValueError("Dataset 5 Post2 must be restored: expected 3,275 missing cells and "
                             f"24 observed zeros, found {blanks} and {zeros}. "
                             "Run prepare_revision3_inputs.py on the original public inputs.")
    finally:
        workbook.close()
    return inputs, checks


def owned_hashes(root, paths):
    root = Path(root).resolve()
    result = {}
    for rel in paths:
        path = (root / rel).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Output escapes result folder.")
        if not path.exists():
            raise FileNotFoundError(f"Missing completed output: {path}")
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        if not files:
            raise ValueError(f"Empty completed output: {path}")
        for file in files:
            if file.is_symlink():
                raise ValueError(f"Unexpected symlink: {file}")
            result[file.relative_to(root).as_posix()] = sha(file)
    return result


def validate_new_output_folder(root):
    """A prior Figure S7 render may share the flat result folder."""
    for path in Path(root).iterdir():
        if path.name not in PLOT_FILES | {"_run.lock"} or not path.is_file() or path.is_symlink():
            raise FileExistsError("Output directory contains unrelated files; choose a new --out folder.")


@contextmanager
def run_lock(root):
    """OS locks are released after process termination, including interruption."""
    with (root / "_run.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another process is using this result folder.") from exc
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another process is using this result folder.") from exc
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


def worker(payload_path):
    payload = read_json(payload_path)
    if runtime(strict=True) != payload["identity"]["runtime"]:
        raise RuntimeError("Worker environment changed.")
    if code_identity() != payload["identity"]["code"]:
        raise RuntimeError("Code changed before the worker started.")
    inputs = {k: Path(v) for k, v in payload["inputs"].items()}
    if set(inputs) != set(KEYS) or {k: sha(v) for k, v in inputs.items()} != payload["identity"]["inputs"]:
        raise RuntimeError("Worker inputs changed.")
    job = payload["job"]
    planned = {j["id"]: j for j in build_plan()}
    if planned.get(job.get("id")) != job:
        raise ValueError("Worker received an unknown or altered job.")
    root = Path(payload["root"])
    context = {"primary_compare_root": str(root / "Compare_method_v1")}
    if job["id"].startswith("core:"):
        import core_jobs
        result = core_jobs.execute_job(job, inputs, root, context)
    else:
        import sensitivity_jobs
        result = sensitivity_jobs.execute_job(job, inputs, root, context)
    write_json(payload["worker_receipt"], {"status": "completed", "result": result,
                                           "completed_utc": utc()})


def run_child(payload, log_path):
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(payload)]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1,
                                   start_new_session=os.name != "nt", env=os.environ.copy())
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                if os.name == "nt":
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        process.kill()
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
    if code:
        raise RuntimeError(f"Job stopped with exit={code}; see {log_path}")


def write_summary(root):
    import pandas as pd
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter
    sources = {
        "Comparison": ["Compare_method_v1/comparison_all_datasets.csv"],
        "Holdout": ["Holdout_v1/holdout_summary_all_datasets.csv",
                    "Holdout_d3-2_V2/holdout_summary_all_datasets.csv"],
        "Selective_nesting": ["Compare_nest_v1/nesting_comparison_all_pre.csv"],
        "Calibration": ["Metrics_Calibration_v1/calibration_primary_metrics_all_datasets.csv"],
        "Sensitivity": ["Compare_sensitivity_v1/comparison_all_scenarios.csv"],
    }
    frames = {}
    for name, files in sources.items():
        frame = pd.concat([pd.read_csv(root / file) for file in files], ignore_index=True)
        if not set(frame["dataset"].astype(str)).issubset(KEYS):
            raise ValueError(f"Unexpected cohort in {name}")
        frames[name] = frame
    expected_rows = {"Comparison": 9, "Holdout": 9, "Selective_nesting": 12,
                     "Calibration": 54, "Sensitivity": 24}
    if {k: len(v) for k, v in frames.items()} != expected_rows:
        raise ValueError("Summary has an incomplete or duplicated set of analysis rows.")
    notes = pd.DataFrame([
        ("Scope", "Scale-aware sensitivity analyses of Datasets 3–5 only; main results are unchanged."),
        ("Comparison", "Patient-averaged OOF predictions; ΔAUC = non-nested minus nested."),
        ("Intervals", "Conditional bootstrap intervals from fixed predictions; not full model-development uncertainty."),
        ("Holdout", "25 stratified 70:30 splits; descriptive median and IQR, not a confidence interval."),
        ("Selective nesting", "Diagnostic hybrid procedures, not fully leakage-free performance estimates."),
        ("Sensitivity", "KNN and expanded grids for 3 Pre matrices; k≤30 and outer3 for all 9 matrices."),
        ("Tests", "No six-cohort test is computed from this three-cohort public sensitivity run."),
        ("Reference", "Compare against the released Table S10/Figure S7 aggregate values."),
    ], columns=["Item", "Description"])
    destination = root / "Revision3_results.xlsx"
    temporary = root / "Revision3_results.tmp.xlsx"
    with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
        notes.to_excel(writer, sheet_name="README", index=False)
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name, index=False)
        for sheet in writer.book:
            sheet.freeze_panes = "B2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="26445C")
            for col in range(1, sheet.max_column + 1):
                sheet.column_dimensions[get_column_letter(col)].width = min(45, max(13, len(str(sheet.cell(1, col).value)) + 2))
            for row in sheet.iter_rows(min_row=2):
                for cell in row:
                    if isinstance(cell.value, float):
                        cell.number_format = "0.000"
        writer.book["README"].column_dimensions["B"].width = 110
    os.replace(temporary, destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Read-only input checks; no model fitting")
    mode.add_argument("--run", action="store_true", help="Run all analyses (also the default)")
    mode.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    identity_runtime = runtime(strict=not args.check)
    inputs, checks = validate_inputs(args.input_dir)
    jobs = build_plan()
    code = code_identity()
    print(f"Scope: Datasets 3–5, 9 matrices, {len(jobs)} jobs; no private-data inputs.", flush=True)
    if args.check:
        print(json.dumps({"status": "validated_no_model_fits", "runtime": identity_runtime,
                          "inputs": checks, "jobs": len(jobs)}, indent=2))
        return
    root = args.out.expanduser().resolve()
    if root == args.input_dir.expanduser().resolve():
        raise ValueError("Input and output directories must differ.")
    root.mkdir(parents=True, exist_ok=True)
    identity = {"scope": "public_D3_D4_D5_scale_aware", "runtime": identity_runtime,
                "code": code, "inputs": {k: sha(v) for k, v in inputs.items()}}
    with run_lock(root):
        receipt_path = root / "RUN_RECEIPT.json"
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt.get("identity") != identity or receipt.get("jobs") != jobs:
                raise RuntimeError("Inputs, code or environment differ from this saved run. "
                                   "Keep the previous results and choose a new --out folder.")
        else:
            validate_new_output_folder(root)
            receipt = {"status": "prepared", "started_utc": utc(), "identity": identity,
                       "jobs": jobs, "inputs": checks}
            write_json(receipt_path, receipt)
        state = root / "_run"
        state.mkdir(exist_ok=True)
        completed = set()
        receipt.update(status="running", updated_utc=utc())
        write_json(receipt_path, receipt)
        try:
            for index, job in enumerate(jobs, 1):
                if code_identity() != code:
                    raise RuntimeError("Code changed during execution.")
                name = job["id"].replace(":", "__")
                saved = state / f"{name}.json"
                record = read_json(saved) if saved.exists() else {}
                if record.get("status") == "completed":
                    if record.get("identity") != identity or record.get("job") != job:
                        raise RuntimeError(f"Saved job identity changed: {job['id']}")
                    if owned_hashes(root, job["owned_paths"]) != record.get("outputs"):
                        raise RuntimeError(f"Saved outputs changed: {job['id']}")
                    completed.add(job["id"])
                    print(f"[{index}/{len(jobs)}] Already complete: {job['id']}", flush=True)
                    continue
                if not set(job.get("depends_on", [])).issubset(completed):
                    raise RuntimeError("A required upstream job is incomplete.")
                attempt = int(record.get("attempt", 0)) + 1
                for rel in job["owned_paths"]:
                    previous = root / rel
                    if previous.exists():
                        archive = root / "_interrupted" / name / f"attempt_{attempt:03d}" / rel
                        archive.parent.mkdir(parents=True, exist_ok=True)
                        if archive.exists():
                            raise FileExistsError(archive)
                        previous.rename(archive)
                record = {"status": "running", "job": job, "identity": identity,
                          "attempt": attempt, "started_utc": utc()}
                write_json(saved, record)
                payload_path = state / f"{name}.payload.json"
                worker_receipt = state / f"{name}.worker.json"
                worker_receipt.unlink(missing_ok=True)
                write_json(payload_path, {"job": job, "root": str(root), "identity": identity,
                                         "inputs": {k: str(v) for k, v in inputs.items()},
                                         "worker_receipt": str(worker_receipt)})
                print(f"\n[{index}/{len(jobs)}] Running: {job['id']}", flush=True)
                try:
                    run_child(payload_path, state / f"{name}.attempt{attempt:03d}.log")
                    if read_json(worker_receipt).get("status") != "completed":
                        raise RuntimeError("Missing worker completion record.")
                    record.update(status="completed", completed_utc=utc(),
                                  outputs=owned_hashes(root, job["owned_paths"]))
                    write_json(saved, record)
                    completed.add(job["id"])
                except BaseException as exc:
                    record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                  error=str(exc), updated_utc=utc())
                    write_json(saved, record)
                    raise
                receipt.update(completed_jobs=len(completed), updated_utc=utc())
                write_json(receipt_path, receipt)
            summary = write_summary(root)
            receipt.update(status="completed", completed_jobs=len(jobs), completed_utc=utc(),
                           summary_file=summary.name, summary_sha256=sha(summary))
            write_json(receipt_path, receipt)
            print(f"\nCompleted. Summary: {summary}", flush=True)
        except BaseException as exc:
            receipt.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                           error=str(exc), completed_jobs=len(completed), updated_utc=utc())
            write_json(receipt_path, receipt)
            print("Completed jobs are retained. Re-run the same command to resume.", flush=True)
            raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
