"""Numerical and provenance checks for read-only revision-2 reporting."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import Revision2_reporting as r


def save_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root, keys=("dataset3_pre",), knn=False):
    """Small complete saved-output tree, using synthetic patient identifiers."""
    main, compare = root / "main", root / "compare"
    kn, sens = root / "knn", root / "sensitivity"
    manifest, comparison_manifest, kn_manifest, sensitivity_manifest = [], [], [], []
    for key in keys:
        nested = main / "main/primary" / key
        nonnested = compare / "datasets" / key
        write_unit(nested, key, True)
        write_unit(nonnested, key, False, source=nested)
        manifest.append({"dataset": key, "scenario": "primary", "output_dir": str(nested)})
        comparison_manifest.append({"dataset": key, "status": "completed", "output_dir": str(nonnested)})
        if knn:
            k = kn / "supplement/sensitivity/knn" / key
            s = sens / "datasets/knn" / key
            write_unit(k, key, True, scenario="knn")
            write_unit(s, key, False, source=k, scenario="knn")
            kn_manifest.append({"dataset": key, "scenario": "knn", "output_dir": str(k)})
            sensitivity_manifest.append({"dataset": key, "scenario": "knn", "status": "completed", "output_dir": str(s)})
    for path, rows in [(main, manifest), (compare, comparison_manifest), (kn, kn_manifest), (sens, sensitivity_manifest)]:
        if rows:
            pd.DataFrame(rows).to_csv(path / "run_manifest.csv", index=False)
    return main, compare, kn, sens


def write_unit(path, key, nested, source=None, scenario="primary"):
    path.mkdir(parents=True)
    ids = ["001", "002", "003", "004"]
    y = [0, 0, 1, 1]
    probabilities = [[.1, .8, .2, .7], [.1, .2, .8, .9]] if nested else [[.1, .2, .8, .9]] * 2
    preds = pd.DataFrame([{"id": pid, "y_true": label, "repeat": repeat, "outer_fold": i % 2 + 1, "p": p}
                          for repeat, ps in enumerate(probabilities, 1)
                          for i, (pid, label, p) in enumerate(zip(ids, y, ps))])
    oof = pd.DataFrame({"id": ids, "y_true": y, "p_oof": np.mean(probabilities, axis=0), "oof_count": 2})
    repeats = pd.DataFrame([{"dataset": key, "repeat": i, "n": 4, "positive": 2, "negative": 2, "roc_auc": r.auc(y, ps)}
                            for i, ps in enumerate(probabilities, 1)])
    metrics = {"dataset": key, "n_samples": 4, "n_positive": 2, "n_negative": 2}
    metrics.update({"roc_auc_oof": .875} if nested else {
        "nested_roc_auc": .875, "nonnested_roc_auc": 1., "delta_auc_nonnested_minus_nested": .125,
        "nested_average_precision": .9, "nonnested_average_precision": 1., "nested_brier": .2, "nonnested_brier": .025,
        "primary_nested_roc_auc": .875, "primary_nonnested_roc_auc": 1., "primary_delta_auc_nonnested_minus_nested": .125})
    settings = {"outer_repeats": 2, "outer_splits_used": 2, "random_state": 42,
                "scenario": {"name": scenario}}
    if not nested:
        settings = {"main_run_receipt_sha256" if scenario == "primary" else "nested_source_receipt_sha256": r.sha256(source / "run_receipt.json"),
                    "selection_and_evaluation_cv": {"splits": 2, "repeats": 2, "random_state": 42}}
    save_json(path / "settings_used.json", settings)
    for filename, frame in [("metrics_summary.csv", pd.DataFrame([metrics])), ("oof_predictions.csv", oof),
                            ("outer_predictions_long.csv", preds), ("repeat_metrics.csv", repeats)]:
        frame.to_csv(path / filename, index=False)
    if nested:
        pd.DataFrame({"feature": ["synthetic_1", "synthetic_2"], "included_in_analysis": [True, False],
                      "n_missing_after_rules": [0, 4]}).to_csv(path / "feature_manifest.csv", index=False)
    receipt = {"status": "completed", "dataset": key, "scenario": scenario, "script_sha256": "synthetic_producer",
               "input_sha256": f"synthetic_input_{key}", "id_column": "id"}
    if not nested:
        receipt["method"] = "fully_non_nested" if scenario == "primary" else "fully_non_nested_sensitivity_compare"
        receipt["output_sha256"] = {p.name: r.sha256(p) for p in path.iterdir()}
    save_json(path / "run_receipt.json", receipt)


class Revision2ReportingTests(unittest.TestCase):
    def test_full_and_subset_omission_values(self):
        table = pd.DataFrame({"dataset": r.PRE, "delta_auc": [.1, .2, .3, .4, .5, .6]})
        full = r.omission_summaries(table)
        self.assertAlmostEqual(full.iloc[0].mean_delta, .35)
        self.assertAlmostEqual(full.iloc[0].median_delta, .35)
        self.assertEqual(full.iloc[0].source_scope, "all_six_main_cohorts")
        self.assertAlmostEqual(full.loc[full.summary == "lcms_only"].iloc[0].mean_delta, .3)
        subset = r.omission_summaries(table.iloc[[2, 5]])
        self.assertEqual(subset.iloc[0].n_retained, 2)
        self.assertAlmostEqual(subset.iloc[0].mean_delta, .45)
        self.assertEqual(subset.iloc[0].source_scope, "available_subset")
        self.assertEqual(len(subset.loc[subset.summary == "omit_one"]), 2)

    def test_one_cohort_omission_is_unavailable_not_zero(self):
        data = pd.DataFrame({"dataset": ["dataset3_pre"], "delta_auc": [.2]})
        omission = r.omission_summaries(data).iloc[1]
        self.assertEqual(omission.status, "no_retained_cohorts")
        self.assertTrue(np.isnan(omission.mean_delta))

    def test_synthetic_saved_outputs_pair_repeats_and_preserve_main_auc(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            m, c, _, _ = fixture(root)
            receipt = r.build(m, c, root / "out", expected_repeats=2)
            self.assertEqual(receipt["primary_scope"], "available_subset")
            self.assertEqual(receipt["available_main_pre"], ["dataset3_pre"])
            repeats = pd.read_csv(root / "out/Paired_repeats.csv")
            np.testing.assert_allclose(repeats.primary_delta_auc, [.5, 0])
            main = pd.read_csv(root / "out/Individual_main_pre.csv")
            self.assertAlmostEqual(main.iloc[0].delta_auc, .125)
            self.assertNotEqual(main.iloc[0].delta_auc, repeats.primary_delta_auc.mean())
            for name, sha in receipt["output_sha256"].items():
                self.assertEqual(sha, r.sha256(root / "out" / name))
            self.assertTrue((root / "out/Revision2_reporting.xlsx").is_file())

    def test_native_knn_subset_exports_expected_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            m, c, k, s = fixture(root, knn=True)
            receipt = r.build(m, c, root / "out", k, s, expected_repeats=2)
            self.assertEqual(receipt["knn_scope"], "available_subset")
            table = pd.read_csv(root / "out/KNN_preprocessing.csv")
            self.assertAlmostEqual(table.iloc[0].knn_delta_auc, .125)
            self.assertEqual(table.iloc[0].sensitivity_setting, "KNN preprocessing")

    def test_reject_duplicate_repeat_and_label_mismatch(self):
        for kind in ["duplicate", "label"]:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                m, c, _, _ = fixture(root)
                unit = m / "main/primary/dataset3_pre"
                file = unit / ("repeat_metrics.csv" if kind == "duplicate" else "outer_predictions_long.csv")
                frame = pd.read_csv(file, dtype={"id": str})
                if kind == "duplicate":
                    frame = pd.concat([frame, frame.iloc[[0]]])
                else:
                    frame.loc[0, "y_true"] = 1
                frame.to_csv(file, index=False)
                with self.assertRaisesRegex(ValueError, "Duplicate|label mismatch"):
                    r.build(m, c, root / "out", expected_repeats=2)
                self.assertFalse((root / "out").exists())

    def test_reject_nonfinite_probability(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            m, c, _, _ = fixture(root)
            file = m / "main/primary/dataset3_pre/oof_predictions.csv"
            frame = pd.read_csv(file, dtype={"id": str})
            frame.loc[0, "p_oof"] = np.inf
            frame.to_csv(file, index=False)
            with self.assertRaisesRegex(ValueError, "Invalid numeric"):
                r.build(m, c, root / "out", expected_repeats=2)

    def test_true_absence_allowed_but_declared_empty_and_missing_outputs_fail(self):
        for kind in ["empty", "missing"]:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                m, c, _, _ = fixture(root)
                file = m / "main/primary/dataset3_pre/repeat_metrics.csv"
                if kind == "empty":
                    pd.read_csv(file).iloc[:0].to_csv(file, index=False)
                else:
                    file.unlink()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    r.build(m, c, root / "out", expected_repeats=2)

    def test_reject_missing_comparison_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            m, c, _, _ = fixture(root, keys=("dataset3_pre", "dataset4_pre"))
            pd.read_csv(c / "run_manifest.csv").iloc[:1].to_csv(c / "run_manifest.csv", index=False)
            with self.assertRaisesRegex(ValueError, "inventories differ"):
                r.build(m, c, root / "out", expected_repeats=2)

    def test_reject_changed_hashed_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            m, c, _, _ = fixture(root)
            file = c / "datasets/dataset3_pre/repeat_metrics.csv"
            with file.open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                r.build(m, c, root / "out", expected_repeats=2)

    def test_reject_stale_source_link(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            m, c, _, _ = fixture(root)
            file = m / "main/primary/dataset3_pre/run_receipt.json"
            data = json.loads(file.read_text())
            data["synthetic_change"] = True
            save_json(file, data)
            with self.assertRaisesRegex(ValueError, "source-receipt mismatch"):
                r.build(m, c, root / "out", expected_repeats=2)

    def test_reject_duplicate_main_cohort_and_malformed_delta(self):
        for frame in [pd.DataFrame({"dataset": ["dataset3_pre"] * 2, "delta_auc": [.1, .2]}),
                      pd.DataFrame({"dataset": ["dataset3_pre"], "delta_auc": ["bad"]}),
                      pd.DataFrame({"dataset": ["dataset3_pre"], "delta_auc": [np.nan]})]:
            with self.assertRaises(ValueError):
                r.omission_summaries(frame)


if __name__ == "__main__":
    unittest.main()
