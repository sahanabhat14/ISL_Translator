"""Real model/IPC check without camera or audio."""
import json
import os
from pathlib import Path
import unittest

import numpy as np
from inference_worker import ModelProcess

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "v1.0.0"


@unittest.skipUnless((MODEL / "isl_lstm.keras").exists() and
                     (ROOT / "models/hand_landmarker.task").exists(), "Needs the release and hand models")
class InferenceProcessTests(unittest.TestCase):
    def test_saved_model_runs_in_its_own_process(self):
        self.check_model(MODEL)

    @unittest.skipUnless((ROOT / "models/15_signs/isl_lstm.keras").exists(), "Needs 15-sign candidate")
    def test_fifteen_sign_candidate_runs_in_its_own_process(self):
        self.check_model(ROOT / "models/15_signs")

    def check_model(self, model_dir):
        report = json.loads((model_dir / "report.json").read_text())
        paths = [ROOT / p for p in report["split"]["test"]]
        inputs = (np.stack([np.load(p, allow_pickle=False) for p in paths])
                  if all(p.exists() for p in paths) else np.zeros((1, 30, 126), np.float32))
        worker = ModelProcess(model_dir)
        try:
            self.assertNotEqual(worker.process.pid, os.getpid())
            predictions = worker(inputs)
            self.assertEqual(predictions.shape, (len(inputs), len(report["labels"])))
            self.assertTrue(np.isfinite(predictions).all())
            np.testing.assert_allclose(predictions.sum(axis=1), 1, atol=1e-5)
            if len(inputs) == len(paths):
                actual = [report["labels"].index(p.parent.name) for p in paths]
                accuracy = float(np.mean(predictions.argmax(1) == actual))
                self.assertAlmostEqual(accuracy, report["test_accuracy"], places=5)
            np.testing.assert_allclose(worker(inputs[:1]), predictions[:1], atol=1e-5)
            tracker = worker.hand_tracker()
            self.assertEqual(tracker.detect_for_video(np.zeros((240, 320, 3), np.uint8), 0).hand_landmarks, [])
            tracker.close()
            self.assertEqual(worker.hand_tracker().detect_for_video(np.zeros((240, 320, 3), np.uint8), 0).hand_landmarks, [])
        finally:
            worker.close()
        self.assertFalse(worker.process.is_alive())


if __name__ == "__main__":
    unittest.main()
