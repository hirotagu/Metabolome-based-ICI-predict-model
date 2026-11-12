"""Fast scope/preprocessing checks; no statistical models are fitted."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("public_r3_runner", ROOT / "run_revision3_6_6.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Revision3RunnerTests(unittest.TestCase):
    def test_jobs_are_public_affected_matrices_only(self):
        jobs = runner.build_plan()
        self.assertEqual(len(jobs), 94)
        self.assertEqual({j["dataset"] for j in jobs if "dataset" in j}, set(runner.KEYS))
        self.assertFalse(any(j["id"].startswith(("reuse:", "report:")) for j in jobs))
        nested = [j for j in jobs if j.get("kind") == "sensitivity_nested"]
        self.assertEqual(len(nested), 24)
        self.assertEqual({j["scenario"] for j in nested}, {"knn", "kmax30", "outer3", "reggrid"})

    def test_log10_input_preserves_negative_zero_and_training_only_imputation(self):
        import numpy as np
        import pandas as pd
        import ICI_predict_v3 as engine
        columns = ["test_metabolite", "Kyn/Trp (log10)"]
        rules = engine.resolve_feature_rules(columns, "dataset4")
        train = pd.DataFrame({columns[0]: [-1., 0., 2., np.nan],
                              columns[1]: [-3., -1., 1., np.nan]})
        fitted = engine.MetaboPreprocessor(columns, feature_rules=rules).fit(train)
        self.assertEqual(rules[columns[0]].transform, "identity")
        self.assertAlmostEqual(fitted._impute_values_raw[columns[0]], -1 - np.log10(2))
        self.assertEqual(fitted._impute_values_raw[columns[1]], -1.)
        before = fitted._mean_.copy()
        heldout = pd.DataFrame({columns[0]: [-8., np.nan, 0.], columns[1]: [0., np.nan, 3.]})
        transformed = fitted.transform(heldout)
        expected_values = heldout.fillna(fitted._impute_values_raw)
        expected = (expected_values - fitted._mean_) / fitted._std_
        np.testing.assert_allclose(transformed, expected.to_numpy())
        pd.testing.assert_series_equal(before, fitted._mean_)

    def test_fast_plot_can_precede_full_run_without_overwriting_other_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in runner.PLOT_FILES:
                (root / name).write_text("test fixture")
            runner.validate_new_output_folder(root)
            (root / "unrelated.xlsx").write_text("do not overwrite")
            with self.assertRaises(FileExistsError):
                runner.validate_new_output_folder(root)


if __name__ == "__main__":
    unittest.main()
