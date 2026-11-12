"""Figure S7 data-contract checks; no model fitting or figure rendering."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("r3_public_plot", ROOT / "plot_revision3_scale_sensitivity.py")
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


class FigureS7InputTests(unittest.TestCase):
    def setUp(self):
        self.data, self.provenance = plot.load_data(ROOT / "figure_data" / "FigureS7_data.csv")

    def test_published_coverage_and_no_patient_fields(self):
        self.assertEqual(self.data.groupby("panel").size().to_dict(), {"A": 6, "B": 12, "C": 9, "D": 6})
        self.assertFalse(set(self.data.columns) & {"id", "y_true", "p_oof", "patient_id"})
        self.assertFalse(self.provenance["individual_level_data_included"])

    def test_duplicate_panel_entry_rejected(self):
        self.data.iloc[1] = self.data.iloc[0]
        with self.assertRaisesRegex(ValueError, "33 unique"):
            plot.validate_data(self.data)

    def test_mismatched_counts_rejected(self):
        self.data.loc[0, "n_samples"] += 1
        with self.assertRaisesRegex(ValueError, "cohort counts"):
            plot.validate_data(self.data)

    def test_invalid_holdout_interval_rejected(self):
        ix = self.data.index[self.data.panel.eq("C")][0]
        self.data.loc[ix, "original_q1"] = self.data.loc[ix, "original"] + 0.01
        with self.assertRaisesRegex(ValueError, "Invalid holdout IQR"):
            plot.validate_data(self.data)

    def test_wrong_hybrid_baseline_rejected(self):
        ix = self.data.index[self.data.panel.eq("D")][0]
        self.data.loc[ix, "original_reference_auc"] += 0.01
        with self.assertRaisesRegex(ValueError, "Hybrid reference"):
            plot.validate_data(self.data)

    def test_altered_source_checksum_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "FigureS7_data.csv"
            dest.write_bytes((ROOT / "figure_data" / dest.name).read_bytes() + b"\n")
            (Path(tmp) / "FigureS7_source.json").write_text(json.dumps(self.provenance))
            with self.assertRaisesRegex(ValueError, "hash differs"):
                plot.load_data(dest)


if __name__ == "__main__":
    unittest.main()
