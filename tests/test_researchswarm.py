import unittest
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from researchswarm import ExecutionStatus, ResearchSwarmEntrypoint


class ResearchSwarmEntrypointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        source_db = Path(__file__).resolve().parents[1] / "AI-Memory" / "memory.db"
        self.memory_db_path = Path(self.tempdir.name) / "memory.db"
        shutil.copy2(source_db, self.memory_db_path)
        self.entrypoint = ResearchSwarmEntrypoint(memory_db_path=self.memory_db_path)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_text_based_task_routes_to_digital_execution(self) -> None:
        report = self.entrypoint.route("Summarize the run log and draft a report")

        self.assertEqual(report.domain, "text-based")
        self.assertEqual(report.status, ExecutionStatus.DIGITAL_EXECUTION)
        self.assertTrue(report.digital_work)
        self.assertFalse(report.human_handoff)

    def test_human_action_task_requires_handoff(self) -> None:
        report = self.entrypoint.route("Visit the lab and photograph the GPU rack")

        self.assertEqual(report.domain, "human-action")
        self.assertEqual(report.status, ExecutionStatus.HUMAN_HANDOFF_REQUIRED)
        self.assertFalse(report.digital_work)
        self.assertTrue(report.human_handoff)

    def test_hybrid_task_splits_digital_and_human_work(self) -> None:
        report = self.entrypoint.route("Draft an inspection checklist and then visit the lab to inspect the rack")

        self.assertEqual(report.domain, "hybrid")
        self.assertEqual(report.status, ExecutionStatus.HYBRID_WORKFLOW)
        self.assertTrue(report.digital_work)
        self.assertTrue(report.human_handoff)
        self.assertIn("Draft an inspection checklist", report.digital_work[0])
        self.assertIn("visit the lab to inspect the rack", report.human_handoff[0])

    def test_research_task_surfaces_repo_commands(self) -> None:
        report = self.entrypoint.route("Prepare the data and run a baseline training experiment")

        self.assertIn("uv run prepare.py", report.suggested_commands)
        self.assertIn("uv run train.py", report.suggested_commands)

    def test_unknown_task_requests_clarification(self) -> None:
        report = self.entrypoint.route("Handle this somehow")

        self.assertEqual(report.status, ExecutionStatus.CLARIFICATION_REQUIRED)
        self.assertTrue(report.human_handoff)
        self.assertTrue(report.next_action)
        self.assertEqual(report.digital_work, ["Clarify the digital scope before attempting execution."])

    def test_research_task_executes_primary_workflow(self) -> None:
        report = self.entrypoint.execute("Prepare the data and run a baseline training experiment overnight")

        self.assertEqual(report.executor_name, "research-workflow")
        self.assertIn("small but real LLM training setup", report.execution_artifact)
        self.assertIn("uv run train.py", report.execution_artifact)

    @patch("researchswarm.subprocess.run")
    @patch("researchswarm.shutil.which")
    def test_research_task_can_launch_prepare_and_train_with_explicit_flags(self, mock_which, mock_run) -> None:
        mock_which.side_effect = lambda name: "uv" if name == "uv" else None
        mock_run.side_effect = [
            type("Completed", (), {"returncode": 0, "stdout": "prepare ok\nready", "stderr": ""})(),
            type("Completed", (), {"returncode": 0, "stdout": "train ok\nval_bpb: 0.99", "stderr": ""})(),
        ]

        report = self.entrypoint.execute(
            "Prepare the data and run a baseline training experiment overnight",
            run_prepare=True,
            run_train=True,
        )

        self.assertEqual(report.executor_name, "research-workflow")
        self.assertIn("Executed Actions", report.execution_artifact)
        self.assertIn("uv run prepare.py", report.execution_artifact)
        self.assertIn("uv run train.py", report.execution_artifact)
        self.assertEqual(mock_run.call_count, 2)

    def test_summarization_executor_uses_supplied_input(self) -> None:
        report = self.entrypoint.execute(
            "Summarize the results",
            input_text="Run one improved validation. Run two regressed slightly. Run three crashed on OOM.",
        )

        self.assertEqual(report.executor_name, "summarizer")
        self.assertIn("Run one improved validation.", report.execution_artifact)

    def test_file_analysis_executor_reads_file(self) -> None:
        readme_path = Path(__file__).resolve().parents[1] / "README.md"
        report = self.entrypoint.execute("Analyze this file", file_path=str(readme_path))

        self.assertEqual(report.executor_name, "file-analyzer")
        self.assertIn("README.md", report.execution_artifact)

    def test_hybrid_task_executes_digital_slice_only(self) -> None:
        report = self.entrypoint.execute(
            "Draft a report and then visit the lab to verify the rack",
            input_text="The rack remained stable during the last test window.",
        )

        self.assertEqual(report.status, ExecutionStatus.HYBRID_WORKFLOW)
        self.assertEqual(report.executor_name, "report-generator")
        self.assertTrue(report.human_handoff)
        self.assertIn("Draft a report", report.digital_work[0])
        self.assertIn("visit the lab to verify the rack", report.human_handoff[0])


if __name__ == "__main__":
    unittest.main()