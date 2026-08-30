"""Tests for inference runtime selection."""

import unittest
import onnxruntime as ort
from drivecv.perception.runtime import create_ort_session


class TestRuntime(unittest.TestCase):
    def test_cpu_provider_available(self):
        self.assertIn("CPUExecutionProvider", ort.get_available_providers())

    def test_create_ort_session_signature(self):
        self.assertTrue(callable(create_ort_session))


if __name__ == "__main__":
    unittest.main()
