"""Tests for process-local NCP DFlash telemetry state."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ncp_olmo_eval.vllm_plugin.ncp_dflash_state import append_telemetry, close_telemetry


class TestDFlashTelemetry(unittest.TestCase):
    def tearDown(self) -> None:
        close_telemetry()

    def test_buffered_records_are_flushed_on_close(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            with patch.dict(os.environ, {"CONCEPTLM_DFLASH_TELEMETRY_PATH": str(path)}):
                append_telemetry("proposal_batch", proposed_tokens=4)
                append_telemetry("verification", accepted_tokens=3)
                close_telemetry()

            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(
                [record["event"] for record in records], ["proposal_batch", "verification"]
            )
            self.assertEqual(records[0]["proposed_tokens"], 4)
            self.assertEqual(records[1]["accepted_tokens"], 3)

    def test_switching_paths_flushes_the_previous_stream(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jsonl"
            second = Path(directory) / "second.jsonl"
            with patch.dict(os.environ, {"CONCEPTLM_DFLASH_TELEMETRY_PATH": str(first)}):
                append_telemetry("first")
            with patch.dict(os.environ, {"CONCEPTLM_DFLASH_TELEMETRY_PATH": str(second)}):
                append_telemetry("second")
                close_telemetry()

            self.assertEqual(json.loads(first.read_text())["event"], "first")
            self.assertEqual(json.loads(second.read_text())["event"], "second")

    def test_flush_interval_persists_records_before_process_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            with patch.dict(
                os.environ,
                {
                    "CONCEPTLM_DFLASH_TELEMETRY_PATH": str(path),
                    "CONCEPTLM_DFLASH_TELEMETRY_FLUSH_INTERVAL": "2",
                },
            ):
                append_telemetry("first")
                self.assertEqual(path.read_text(), "")
                append_telemetry("second")
                records = [json.loads(line) for line in path.read_text().splitlines()]

            self.assertEqual([record["event"] for record in records], ["first", "second"])


if __name__ == "__main__":
    unittest.main()

