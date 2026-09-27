"""Aggregate-only renderer guards; optional matplotlib tests skip if absent."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
if HAS_MATPLOTLIB:
    import Revision2_figures as figures


def fixture(root, keys=("dataset3_pre",)):
    root.mkdir()
    rows = []
    for key in keys:
        for method in figures.METHODS:
            for bin_number in (1, 2):
                rows.append({"dataset": key, "method": method, "probability_variant": "raw", "bin": bin_number,
                             "n": 4, "n_positive": 2, "predicted_mean": .2 if bin_number == 1 else .8,
                             "observed_rate": .5, "observed_rate_wilson95_low": .15003898915214947,
                             "observed_rate_wilson95_high": .8499610108478506, "dataset_event_rate": .5})
    pd.DataFrame(rows).to_csv(root / "calibration_wilson_bins.csv", index=False)
    scope = "full_six" if len(keys) == 6 else "available_subset"
    pd.DataFrame([{"dataset": key, "status": "included" if key in keys else "not_provided", "coverage_scope": scope}
                  for key in figures.PRE]).to_csv(root / "coverage.csv", index=False)
    pd.DataFrame([{"dataset": key, "method": method, "producer_sha256": "synthetic"}
                  for key in keys for method in figures.METHODS]).to_csv(root / "provenance.csv", index=False)
    receipt = {"status": "completed", "model_fitting_performed": False, "datasets": list(keys),
               "coverage_scope": scope, "output_sha256": {name: figures.sha256(root / name) for name in figures.SOURCE_FILES}}
    (root / "calibration_receipt.json").write_text(json.dumps(receipt))


def refresh_hash(root, filename):
    p = root / "calibration_receipt.json"
    data = json.loads(p.read_text())
    data["output_sha256"][filename] = figures.sha256(root / filename)
    p.write_text(json.dumps(data))


@unittest.skipUnless(HAS_MATPLOTLIB, "Install requirements-figures.txt for renderer tests")
class FigureTests(unittest.TestCase):
    def test_subset_render_and_coverage_annotation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root / "input")
            table, keys, totals, scope, _ = figures.load_calibration(root / "input")
            fig, *_ = figures.make_figure(table, keys, totals, scope, "DejaVu Sans")
            try:
                self.assertIn("Available subset: 1/6 intended Pre cohorts", [text.get_text() for text in fig.texts])
                self.assertEqual(totals["dataset3_pre"], (8, 4))  # Do not double count methods.
                self.assertEqual(len(fig.axes[0].containers), 2)
            finally:
                figures.plt.close(fig)
            receipt = figures.build(root / "input", root / "out", formats=("png",), dpi=72, font="DejaVu Sans")
            self.assertEqual(receipt["coverage_scope"], "available_subset")
            self.assertFalse(receipt["statistical_estimation_performed"])
            self.assertGreater((root / "out/FigureS2_calibration.png").stat().st_size, 1000)
            with self.assertRaises(FileExistsError):
                figures.build(root / "input", root / "out", formats=("png",), dpi=72, font="DejaVu Sans")

    def test_full_six_uses_two_by_three_without_subset_label(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root / "input", figures.PRE)
            table, keys, totals, scope, _ = figures.load_calibration(root / "input")
            fig, _, _, size = figures.make_figure(table, keys, totals, scope, "DejaVu Sans")
            try:
                self.assertEqual(size, (9.2, 6.2))
                self.assertEqual(len(fig.axes), 6)
                self.assertFalse(fig.texts)
            finally:
                figures.plt.close(fig)

    def test_hash_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "input"
            fixture(root)
            with (root / "calibration_wilson_bins.csv").open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                figures.load_calibration(root)

    def test_counts_duplicate_bins_and_coverage_errors_are_rejected_even_with_updated_hash(self):
        for problem in ("counts", "duplicate", "coverage", "missing_method"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "input"
                fixture(root)
                name = "coverage.csv" if problem == "coverage" else "calibration_wilson_bins.csv"
                frame = pd.read_csv(root / name)
                if problem == "counts":
                    frame.loc[0, "n_positive"] = 3
                elif problem == "duplicate":
                    frame = pd.concat([frame, frame.iloc[[0]]])
                elif problem == "coverage":
                    frame.loc[frame.dataset == "dataset3_pre", "status"] = "not_provided"
                else:
                    frame = frame.loc[frame.method == "fully_nested"]
                frame.to_csv(root / name, index=False)
                refresh_hash(root, name)
                with self.assertRaises(ValueError):
                    figures.load_calibration(root)


if __name__ == "__main__":
    unittest.main()
