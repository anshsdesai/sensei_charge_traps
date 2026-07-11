import unittest
from pathlib import Path

from signed_refit_manifest import (
    MANIFEST_VERSION,
    load_selected_image_files,
    read_manifest,
)


class SignedRefitManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent
        cls.manifest = cls.repo_root / "signed_refit_manifest.csv"
        cls.rows = read_manifest(cls.manifest)

    def test_manifest_version_and_counts(self):
        self.assertEqual({row["manifest_version"] for row in self.rows}, {MANIFEST_VERSION})
        self.assertEqual(len(self.rows), 481)
        self.assertEqual(sum(int(row["selected"]) for row in self.rows), 477)

    def test_selected_temperature_dtph_is_unique(self):
        selected = [row for row in self.rows if int(row["selected"]) == 1]
        keys = [(int(row["temperature_K"]), int(row["dtph"])) for row in selected]
        self.assertEqual(len(keys), len(set(keys)))

    def test_recent_200k_selection(self):
        selected = [
            row for row in self.rows
            if int(row["selected"]) == 1 and int(row["temperature_K"]) == 200
        ]
        excluded = [row for row in self.rows if int(row["selected"]) == 0]
        self.assertEqual({int(row["run_id"]) for row in selected}, set(range(160, 185)))
        self.assertEqual(
            {(int(row["dtph"]), int(row["run_id"])) for row in excluded},
            {(750, 21), (1200, 22), (2000, 23), (3000, 24)},
        )
        self.assertTrue(all(row["exclusion_reason"] for row in excluded))

    def test_loader_resolves_all_selected_files(self):
        files, sha256 = load_selected_image_files(self.manifest, self.repo_root)
        self.assertEqual(len(files), 477)
        self.assertEqual(len(sha256), 64)
        self.assertTrue(all(Path(path).is_file() for path in files))


if __name__ == "__main__":
    unittest.main()
