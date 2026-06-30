import unittest
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from researchswarm import ExecutionStatus, ResearchSwarmEntrypoint


class ResearchSwarmEntrypointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.tempdir.name)
        source_db = Path(__file__).resolve().parents[1] / "AI-Memory" / "memory.db"
        self.memory_db_path = self.repo_root / "memory.db"
        shutil.copy2(source_db, self.memory_db_path)
        self.entrypoint = ResearchSwarmEntrypoint(
            memory_db_path=self.memory_db_path,
            wiki_repo_root=self.repo_root,
        )

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

    def test_loop_engineering_executor_selects_pattern_from_thirdparty_registry(self) -> None:
        report = self.entrypoint.execute("Run a daily triage loop over issues and CI status")

        self.assertEqual(report.executor_name, "loop-engineering")
        self.assertIn("Loop Engineering Activation", report.execution_artifact)
        self.assertIn("selected_pattern_id: daily-triage", report.execution_artifact)
        self.assertIn("thirdparty", report.execution_artifact)

    def test_loop_engineering_executor_preserves_human_handoff_for_hybrid_prompts(self) -> None:
        report = self.entrypoint.execute(
            "Run dependency sweeper triage and then visit the server room to reseat a cable",
        )

        self.assertEqual(report.status, ExecutionStatus.HYBRID_WORKFLOW)
        self.assertEqual(report.executor_name, "loop-engineering")
        self.assertIn("Human Handoff Boundary", report.execution_artifact)
        self.assertTrue(report.human_handoff)

    def test_wiki_ingest_creates_markdown_page_and_updates_index(self) -> None:
        run_log = self.repo_root / "run.log"
        run_log.write_text(
            "baseline experiment\nval_bpb improved from 1.21 to 1.18\noptimizer: adamw\n",
            encoding="utf-8",
        )

        report = self.entrypoint.execute(f"wiki ingest {run_log}")

        self.assertEqual(report.executor_name, "wiki")
        self.assertIn("# Wiki Ingest", report.execution_artifact)
        index_path = self.repo_root / "LLM-Wiki" / "wiki" / "index.md"
        self.assertTrue(index_path.exists())
        self.assertIn("run", index_path.read_text(encoding="utf-8", errors="replace"))

    def test_wiki_query_answers_from_cited_pages(self) -> None:
        source = self.repo_root / "attention_notes.md"
        source.write_text(
            "Sliding-window attention reduced val_bpb by 0.02 on baseline.",
            encoding="utf-8",
        )
        self.entrypoint.execute(f"wiki ingest {source}")

        report = self.entrypoint.execute("wiki query what architecture changes improved val_bpb?")

        self.assertEqual(report.executor_name, "wiki")
        self.assertIn("## Citations", report.execution_artifact)
        self.assertIn("LLM-Wiki/wiki", report.execution_artifact)

    def test_wiki_lint_reports_health_findings(self) -> None:
        source = self.repo_root / "oom.log"
        source.write_text("OOM failure observed during training run.", encoding="utf-8")
        self.entrypoint.execute(f"wiki ingest {source}")

        report = self.entrypoint.execute("wiki lint")

        self.assertEqual(report.executor_name, "wiki")
        self.assertIn("# Wiki Lint", report.execution_artifact)
        self.assertIn("scanned_pages", report.execution_artifact)

    def test_wiki_ingest_adds_related_links_and_backlinks(self) -> None:
        first = self.repo_root / "notes_a.log"
        second = self.repo_root / "notes_b.log"
        first.write_text(
            "architecture sliding window attention improved val_bpb and optimizer adamw was stable",
            encoding="utf-8",
        )
        second.write_text(
            "sliding window attention run observed val_bpb improved with adamw optimizer tweaks",
            encoding="utf-8",
        )

        self.entrypoint.execute(f"wiki ingest {first}")
        self.entrypoint.execute(f"wiki ingest {second}")

        page_one_matches = list((self.repo_root / "LLM-Wiki" / "wiki").rglob("notes-a.md"))
        page_two_matches = list((self.repo_root / "LLM-Wiki" / "wiki").rglob("notes-b.md"))
        self.assertTrue(page_one_matches)
        self.assertTrue(page_two_matches)
        page_one = page_one_matches[0]
        page_two = page_two_matches[0]
        self.assertTrue(page_one.exists())
        self.assertTrue(page_two.exists())
        self.assertIn("## Related Pages", page_one.read_text(encoding="utf-8", errors="replace"))
        self.assertIn("notes-b", page_one.read_text(encoding="utf-8", errors="replace"))
        self.assertIn("notes-a", page_two.read_text(encoding="utf-8", errors="replace"))

    def test_wiki_lint_detects_contradictory_experiment_claims(self) -> None:
        improved = self.repo_root / "exp_improved.log"
        regressed = self.repo_root / "exp_regressed.log"
        improved.write_text("optimizer adamw improved val_bpb from 1.20 to 1.15", encoding="utf-8")
        regressed.write_text("optimizer adamw regressed val_bpb from 1.15 to 1.19", encoding="utf-8")

        self.entrypoint.execute(f"wiki ingest {improved}")
        self.entrypoint.execute(f"wiki ingest {regressed}")
        report = self.entrypoint.execute("wiki lint")

        self.assertIn("contradictory_experiment_claims", report.execution_artifact)
        self.assertNotIn("contradictory_experiment_claims: 0", report.execution_artifact)
        self.assertIn("adamw", report.execution_artifact)
        self.assertIn("severity=high", report.execution_artifact)

    def test_wiki_batch_ingest_supports_folder_in_raw(self) -> None:
        batch_folder = self.repo_root / "LLM-Wiki" / "raw" / "run_logs"
        batch_folder.mkdir(parents=True, exist_ok=True)
        (batch_folder / "a.log").write_text("baseline run val_bpb improved", encoding="utf-8")
        (batch_folder / "b.log").write_text("dataset change regressed val_bpb", encoding="utf-8")

        report = self.entrypoint.execute("wiki ingest --batch LLM-Wiki/raw/run_logs")

        self.assertEqual(report.executor_name, "wiki")
        self.assertIn("# Wiki Batch Ingest", report.execution_artifact)
        self.assertIn("files_ingested: 2", report.execution_artifact)


if __name__ == "__main__":
    unittest.main()