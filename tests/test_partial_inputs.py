"""Subset input regression checks; no model fitting or private data required."""
import argparse
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import ICI_predict as engine
from dataset_selection import FORMAL_PRE_KEYS, KNOWN_DATASET_KEYS, select_available
from prepare_inputs import SHEET_PLAN, prepare_inputs


class PartialInputsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        engine.configure_scenario("primary")

    def tearDown(self):
        engine.configure_scenario("primary")
        self.temporary.cleanup()

    def frame(self, key):
        _, _, n, positive, _, features = next(item for item in SHEET_PLAN if item[1] == key)
        frame = pd.DataFrame(np.ones((n, features)), columns=[f"feature{i}" for i in range(features)])
        frame.insert(0, "category", [1] * positive + [0] * (n - positive))
        frame.insert(0, "id", [f"sample{i}" for i in range(n)])
        return frame

    def workbook(self, sheets):
        path = self.root / "source.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=name, index=False)
        return path

    def prepare(self, source):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare_inputs(source, self.root / "raw")

    def test_noncontiguous_subset_preserves_original_mapping_and_qc(self):
        source = self.workbook({"Sheet3": self.frame("dataset3_pre"), "Sheet7": self.frame("dataset7_pre")})
        qc = self.prepare(source)
        self.assertEqual(qc.dataset.tolist(), ["dataset3_pre", "dataset7_pre"])
        self.assertEqual(qc.n.tolist(), [77, 39])
        self.assertFalse((self.root / "raw/dataset1_pre.xlsx").exists())
        coverage = pd.read_csv(self.root / "raw/input_coverage.csv").set_index("dataset")
        self.assertEqual(len(coverage), 17)
        self.assertEqual(coverage.loc["dataset2_pre", "input_status"], "not_provided")
        self.assertEqual(coverage.loc["dataset7_pre", "input_status"], "available")
        actual = pd.read_excel(self.root / "raw/dataset3_pre.xlsx")
        pd.testing.assert_frame_equal(actual, self.frame("dataset3_pre"), check_dtype=False)

    def test_canonical_sheet_name_and_optional_readme(self):
        source = self.workbook({"README": pd.DataFrame({"note": ["test"]}), "dataset7_pre": self.frame("dataset7_pre")})
        self.assertEqual(self.prepare(source).dataset.tolist(), ["dataset7_pre"])

    def test_duplicate_alias_unknown_and_empty_workbooks_fail(self):
        cases = (
            {"Sheet7": self.frame("dataset7_pre"), "dataset7_pre": self.frame("dataset7_pre")},
            {"Sheet99": self.frame("dataset7_pre")},
            {"README": pd.DataFrame({"note": ["no data"]})},
        )
        for sheets in cases:
            with self.subTest(sheets=list(sheets)):
                with self.assertRaises(ValueError):
                    self.prepare(self.workbook(sheets))
                self.assertFalse((self.root / "raw").exists())

    def test_malformed_input_retains_qc_and_all_or_nothing(self):
        for problem in ("count", "label", "duplicate_id", "non_numeric", "infinity"):
            frame = self.frame("dataset7_pre")
            if problem == "count":
                frame = frame.iloc[:-1]
            elif problem == "label":
                frame.loc[0, "category"] = 7
            elif problem == "duplicate_id":
                frame.loc[1, "id"] = frame.loc[0, "id"]
            elif problem == "non_numeric":
                frame["feature0"] = frame["feature0"].astype(object)
                frame.loc[0, "feature0"] = "malformed"
            else:
                frame.loc[0, "feature0"] = np.inf
            with self.subTest(problem=problem):
                source = self.workbook({"Sheet3": self.frame("dataset3_pre"), "Sheet7": frame})
                with self.assertRaises(ValueError):
                    self.prepare(source)
                self.assertFalse((self.root / "raw").exists())

    def test_existing_inputs_are_not_overwritten(self):
        source = self.workbook({"Sheet7": self.frame("dataset7_pre")})
        self.prepare(source)
        target = self.root / "raw/dataset7_pre.xlsx"
        before = target.read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare(source)
        self.assertEqual(before, target.read_bytes())

    def test_selector_rejects_explicit_missing_unknown_and_duplicate(self):
        self.assertEqual(select_available("ALL", ["dataset7_pre"], FORMAL_PRE_KEYS), ["dataset7_pre"])
        with self.assertRaises(FileNotFoundError):
            select_available(["dataset2_pre"], ["dataset7_pre"])
        for request in (["dataset7_post2"], ["dataset7_pre", "dataset7_pre"]):
            with self.assertRaises(ValueError):
                select_available(request, ["dataset7_pre"])

    def test_profile_defaults_intersect_but_cli_missing_is_error(self):
        specs = {
            key: engine.DatasetSpec(self.root / f"{key}.xlsx", key.split("_")[0], "pre", key)
            for key in ("dataset3_pre", "dataset6_pre", "dataset7_pre")
        }
        for scenario in ("knn", "reggrid"):
            with self.subTest(scenario=scenario):
                engine.configure_scenario(scenario)
                self.assertEqual(engine._intersect_with_global_targets(list(specs)), ["dataset3_pre", "dataset7_pre"])
        engine.configure_scenario("knn")
        scheduled = engine._sensitivity_scenarios(specs, {})
        self.assertEqual([spec.key for spec, _ in scheduled], ["dataset3_pre", "dataset7_pre"])
        self.assertTrue(all(scenario.imputation == "knn" for _, scenario in scheduled))
        engine.configure_cli(argparse.Namespace(scenario="knn", datasets=["dataset2_pre"]))
        with self.assertRaises(FileNotFoundError):
            engine._intersect_with_global_targets(list(specs))

    def test_primary_all_known_inputs_and_sensitivity_no_applicable_inputs(self):
        self.assertEqual(engine._intersect_with_global_targets(["dataset6_pre", "dataset7_pre"]), ["dataset6_pre", "dataset7_pre"])
        engine.configure_scenario("sd")
        self.assertEqual(engine._intersect_with_global_targets(["dataset7_pre"]), [])

    def test_no_applicable_sensitivity_records_status_without_fitting(self):
        key = "dataset7_pre"
        specs = {key: engine.DatasetSpec(self.root / f"{key}.xlsx", "dataset7", "pre", key)}
        engine.configure_scenario("sd")
        with patch.object(engine, "discover_datasets", return_value=specs), \
             patch.object(engine, "OUTPUT_ROOT", self.root / "results"), \
             patch.object(engine, "run_nested_cv") as fit, \
             contextlib.redirect_stdout(io.StringIO()) as stdout:
            engine.main()
            fit.assert_not_called()
        self.assertIn("No supplied inputs apply", stdout.getvalue())
        coverage = pd.read_csv(self.root / "results/All_data_SD/input_coverage.csv")
        self.assertEqual(set(coverage.invocation_status), {"no_applicable_inputs"})
        self.assertFalse((self.root / "results/All_data_SD/run_manifest.csv").exists())

    def test_empty_primary_targets_are_not_reported_completed(self):
        key = "dataset7_pre"
        specs = {key: engine.DatasetSpec(self.root / f"{key}.xlsx", "dataset7", "pre", key)}
        with patch.object(engine, "discover_datasets", return_value=specs), \
             patch.object(engine, "OUTPUT_ROOT", self.root / "results"), \
             patch.object(engine, "PRIMARY_TARGETS", []), \
             patch.object(engine, "run_nested_cv") as fit, \
             contextlib.redirect_stdout(io.StringIO()) as stdout:
            engine.main()
            fit.assert_not_called()
        self.assertIn("No analyses were scheduled", stdout.getvalue())
        self.assertNotIn("completed", stdout.getvalue())
        coverage = pd.read_csv(self.root / "results/All_data_primary/input_coverage.csv")
        self.assertEqual(set(coverage.invocation_status), {"no_analyses_scheduled"})
        self.assertEqual(set(coverage.selection_status), {"not_selected"})
        self.assertFalse((self.root / "results/All_data_primary/run_manifest.csv").exists())

    def test_settings_record_selector_source_hash(self):
        spec = engine.DatasetSpec(self.root / "dataset7_pre.xlsx", "dataset7", "pre", "dataset7_pre")
        settings = engine._scenario_settings_payload(spec, engine._primary_scenario(spec), 5)
        selector_path = Path(settings["dataset_selection_path"])
        self.assertEqual(selector_path.name, "dataset_selection.py")
        self.assertEqual(settings["dataset_selection_sha256"], engine.sha256_file(selector_path))
        self.assertEqual(settings["dataset_selection_sha256"], engine._selector_provenance()["dataset_selection_sha256"])

    def test_discovery_rejects_unknown_duplicate_or_no_inputs(self):
        raw = self.root / "raw"
        raw.mkdir()
        with patch.object(engine, "INPUT_DIR", raw), patch.object(engine, "INPUT_FILES", []):
            with self.assertRaises(RuntimeError):
                engine.discover_datasets()
            unknown = raw / "dataset7_post2.xlsx"
            unknown.touch()
            with self.assertRaises(ValueError):
                engine.discover_datasets()
            unknown.unlink()
            self.frame("dataset7_pre").to_excel(raw / "dataset7_pre.xlsx", index=False)
            self.assertEqual(list(engine.discover_datasets()), ["dataset7_pre"])
            (raw / "duplicate").mkdir()
            self.frame("dataset7_pre").to_excel(raw / "duplicate/dataset7_pre.xlsx", index=False)
            with self.assertRaises(ValueError):
                engine.discover_datasets()

    def test_original_study_map_remains_exact(self):
        self.assertEqual(tuple(item[1] for item in SHEET_PLAN), KNOWN_DATASET_KEYS)
        self.assertEqual(len(FORMAL_PRE_KEYS), 6)
        self.assertNotIn("dataset6_pre", FORMAL_PRE_KEYS)


if __name__ == "__main__":
    unittest.main()
