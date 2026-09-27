"""Partial input coverage must not weaken artifact or inferential contracts."""
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import Compare_method as method
import Compare_nest as nest
import Compare_sensitivity as sensitivity
import Holdout as holdout
import Metrics_Calibration as calibration


PUBLIC_KEYS = tuple(key for key in method.CANONICAL_KEYS if not key.startswith(("dataset1_", "dataset2_")))
PUBLIC_PRE = [f"dataset{n}_pre" for n in (3, 4, 5, 7)]


class DownstreamSelectionTests(unittest.TestCase):
    def test_default_available_inputs_preserve_identifiers(self):
        specs = dict.fromkeys(PUBLIC_KEYS)
        for module in (method, holdout, calibration):
            with self.subTest(module=module.__name__), patch.object(module, "DATASETS_TO_RUN", "ALL"):
                self.assertEqual(module._selected_keys(specs), list(PUBLIC_KEYS))
        with patch.object(nest, "DATASETS_TO_RUN", "ALL"):
            self.assertEqual(nest._selected_keys(specs), PUBLIC_PRE)

    def test_explicit_missing_unknown_and_duplicate_requests_fail(self):
        specs = dict.fromkeys(PUBLIC_KEYS)
        for module in (method, nest, holdout, calibration):
            for request in ("dataset2_pre", ["dataset2_pre"]):
                with self.subTest(module=module.__name__, request=request), patch.object(module, "DATASETS_TO_RUN", request):
                    with self.assertRaises(FileNotFoundError):
                        module._selected_keys(specs)
            for request in ("dataset99_pre", ["dataset3_pre", "dataset3_pre"]):
                with patch.object(module, "DATASETS_TO_RUN", request), self.assertRaises(ValueError):
                    module._selected_keys(specs)
            with patch.object(module, "DATASETS_TO_RUN", "ALL"), self.assertRaises(ValueError):
                module._selected_keys({**specs, "dataset99_pre": None})

    def test_partial_cohorts_never_run_six_cohort_formal_test(self):
        table = pd.DataFrame({"dataset": PUBLIC_PRE, "delta_auc_nonnested_minus_nested": [0.1, 0.2, 0.05, 0.12]})
        inference = method.build_formal_inference(table)
        self.assertEqual(inference.iloc[0]["test"], "not_run")
        self.assertEqual(inference.iloc[0]["missing_datasets"], "dataset1_pre|dataset2_pre")
        self.assertNotIn("p_value", inference)

    def test_six_cohort_test_unchanged_when_all_provided(self):
        table = pd.DataFrame({"dataset": method.FORMAL_PRE_KEYS, "delta_auc_nonnested_minus_nested": [0.1, 0.2, 0.05, 0.12, 0.18, 0.15]})
        inference = method.build_formal_inference(table)
        self.assertTrue((inference["status"] == "completed").all())
        self.assertTrue((inference["n_cohorts"] == 6).all())
        self.assertAlmostEqual(inference.iloc[0]["p_value"], 0.03125)
        self.assertEqual(inference.iloc[0]["inferential_role"], "descriptive_nominal_primary")
        self.assertTrue(inference["p_value_interpretation"].str.contains("descriptive nominal").all())

    def test_sensitivity_scope_and_coverage(self):
        specs = dict.fromkeys(PUBLIC_KEYS)
        configs = sensitivity._all_configs()
        with patch.object(sensitivity, "DATASETS_TO_RUN", "ALL"):
            self.assertEqual(sensitivity._selected_keys(configs["knn"], specs), PUBLIC_PRE)
            self.assertEqual(sensitivity._selected_keys(configs["sd_as_nr"], specs), [])
        with patch.object(sensitivity, "DATASETS_TO_RUN", "dataset2_pre"), self.assertRaises(FileNotFoundError):
            sensitivity._selected_keys(configs["knn"], specs)
        self.assertEqual(sensitivity.KNN_KEYS, method.FORMAL_PRE_KEYS)
        units = [(configs["knn"], key) for key in PUBLIC_PRE]
        coverage = sensitivity._coverage_table(specs, [configs["knn"]], units).set_index("dataset")
        self.assertEqual(coverage.loc["dataset2_pre", "input_status"], "not_provided")
        self.assertEqual(coverage.loc["dataset6_pre", "selection_status"], "not_applicable")
        self.assertEqual(int((coverage["selection_status"] == "selected").sum()), 4)

    def test_missing_selected_completed_output_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame([{"dataset": "dataset3_pre", "status": "completed", "output_dir": str(root / "datasets" / "dataset3_pre")}]).to_csv(root / "run_manifest.csv", index=False)
            with self.assertRaisesRegex(FileNotFoundError, "dataset4_pre"):
                calibration._resolve_completed_dataset_dir(root, "dataset4_pre")
            with self.assertRaisesRegex(FileNotFoundError, "output directory"):
                calibration._resolve_completed_dataset_dir(root, "dataset3_pre")

    def _write_primary_reference(self, root, keys):
        (root / "run_receipt.json").write_text(json.dumps({"status": "completed", "n_failed": 0}))
        (root / "settings_used.json").write_text("{}")
        pd.DataFrame([{"test": "not_run"}]).to_csv(root / "pre_primary_inference.csv", index=False)
        table = pd.DataFrame({"dataset": keys})
        for col in ("n_samples", "n_positive", "n_negative", "nested_roc_auc", "nested_average_precision", "nested_brier", "nonnested_roc_auc", "nonnested_average_precision", "nonnested_brier", "delta_auc_nonnested_minus_nested", "delta_ap_nonnested_minus_nested", "delta_brier_nonnested_minus_nested"):
            table[col] = 1
        table.to_csv(root / "comparison_all_datasets.csv", index=False)
        pd.DataFrame({"dataset": keys, "status": "completed"}).to_csv(root / "run_manifest.csv", index=False)

    def test_partial_primary_reference_accepts_present_outputs_but_not_missing_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_primary_reference(root, PUBLIC_PRE)
            with patch.object(sensitivity, "PRIMARY_COMPARE_RUN_ROOT", root), patch.object(sensitivity, "STRICT_PRIMARY_REFERENCE_HASH", False):
                table, hashes = sensitivity._validate_primary_reference(PUBLIC_PRE)
                self.assertEqual(table["dataset"].tolist(), PUBLIC_PRE)
                self.assertEqual(len(hashes), 5)
                with self.assertRaisesRegex(ValueError, "missing selected"):
                    sensitivity._validate_primary_reference([*PUBLIC_PRE, "dataset2_pre"])

    def test_missing_scenario_output_is_not_treated_as_unavailable_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = replace(sensitivity._all_configs()["knn"], run_root=root)
            pd.DataFrame([{"dataset": "dataset3_pre", "scenario": "knn", "invocation_scenario": "knn", "status": "completed", "output_dir": "unused"}]).to_csv(root / "run_manifest.csv", index=False)
            with self.assertRaisesRegex(ValueError, "missing selected completed datasets"):
                sensitivity._preflight_source_inventory([config], [(config, "dataset4_pre")])


class WilsonCoordinatesTests(unittest.TestCase):
    def test_exact_bin_counts_and_known_wilson_interval(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        p = np.linspace(0.1, 0.8, 8)
        result = calibration._calibration_coordinates(y, p, 2)
        self.assertEqual(result["n"].tolist(), [4, 4])
        self.assertEqual(result["n_positive"].tolist(), [0, 4])
        self.assertEqual(result["observed_rate"].tolist(), [0.0, 1.0])
        self.assertAlmostEqual(result.iloc[0]["observed_rate_wilson95_low"], 0.0)
        self.assertAlmostEqual(result.iloc[0]["observed_rate_wilson95_high"], 0.4898908364545973)
        self.assertAlmostEqual(result.iloc[1]["observed_rate_wilson95_low"], 0.5101091635454027)
        self.assertTrue(result["interval_scope"].str.contains("not simultaneous").all())

    def test_identical_predictions_keep_single_bin(self):
        result = calibration._calibration_coordinates(np.array([0, 1, 1]), np.full(3, 0.5), 10)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["n_positive"], 2)
        self.assertAlmostEqual(result.iloc[0]["observed_rate"], 2 / 3)

    def test_invalid_calibration_input_is_rejected(self):
        for y, p in (([0, 0.5], [0.1, 0.2]), ([0, 1], [0.1, np.nan]), ([0, 1], [0.1]), ([], [])):
            with self.subTest(y=y, p=p), self.assertRaises(ValueError):
                calibration._calibration_coordinates(np.array(y), np.array(p), 2)


if __name__ == "__main__":
    unittest.main()
