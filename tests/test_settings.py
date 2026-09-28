"""
Unit tests for AirControl V2 Settings Management (src/settings_ui.py).

Tests:
1. Default settings fallback.
2. Saving and loading valid configuration from settings.json.
3. Dynamic propagation to config module.
4. Handling missing/corrupted settings file gracefully.
"""

import json
import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import config
from src import settings_ui


class TestSettingsManagement(unittest.TestCase):

    def setUp(self):
        # Backup original settings file if it exists
        self.original_settings_file = settings_ui.SETTINGS_FILE
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_file.close()
        settings_ui.SETTINGS_FILE = self.temp_file.name

        # Snapshot config state
        self.original_config_state = {
            k: getattr(config, k) for k in dir(config) if not k.startswith("__")
        }

    def tearDown(self):
        settings_ui.SETTINGS_FILE = self.original_settings_file
        if os.path.exists(self.temp_file.name):
            try:
                os.remove(self.temp_file.name)
            except Exception:
                pass

        # Restore config state
        for k, v in self.original_config_state.items():
            setattr(config, k, v)

    def test_default_settings_loaded_when_file_missing(self):
        """Verify defaults are loaded when settings.json does not exist."""
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)

        loaded = settings_ui.load_settings()
        self.assertIn("SMOOTHING_FACTOR", loaded)
        self.assertIn("SCROLL_AMOUNT", loaded)
        self.assertIn("DEFAULT_MODE", loaded)

    def test_save_and_load_roundtrip(self):
        """Verify settings dictionary saves and reloads identically."""
        custom_data = {
            "SMOOTHING_FACTOR": 0.42,
            "SCROLL_AMOUNT": 45,
            "DEFAULT_MODE": "MEDIA",
            "FLIP_CAMERA": False,
        }
        success = settings_ui.save_settings(custom_data)
        self.assertTrue(success)

        loaded = settings_ui.load_settings()
        self.assertEqual(loaded["SMOOTHING_FACTOR"], 0.42)
        self.assertEqual(loaded["SCROLL_AMOUNT"], 45)
        self.assertEqual(loaded["DEFAULT_MODE"], "MEDIA")
        self.assertEqual(loaded["FLIP_CAMERA"], False)

        # Also verify live config module updated
        self.assertAlmostEqual(config.SMOOTHING_FACTOR, 0.42)
        self.assertEqual(config.SCROLL_AMOUNT, 45)
        self.assertEqual(config.DEFAULT_MODE, "MEDIA")
        self.assertFalse(config.FLIP_CAMERA)

    def test_corrupted_json_handled_gracefully(self):
        """Verify invalid/corrupted JSON file falls back to defaults without crashing."""
        with open(self.temp_file.name, "w") as f:
            f.write("{invalid json content!@@#}")

        loaded = settings_ui.load_settings()
        self.assertIsInstance(loaded, dict)
        self.assertIn("SMOOTHING_FACTOR", loaded)


if __name__ == "__main__":
    unittest.main()
