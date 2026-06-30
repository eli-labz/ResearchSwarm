# Changelog

All notable changes to ResearchSwarm are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

---

## [Unreleased]

### Planned
- GPU memory usage reporting in experiment logs
- Automatic plateau detection to skip similar experiments
- Web dashboard for real-time overnight monitoring

---

## [0.1.0] — 2026-03-15

### Added
- **Digital Cognitive Labor router** — classifies any natural-language task into `text-based`, `human-action`, or `hybrid`
- **AI memory store** (`researchswarm_memory.py`) — SQLite-backed persistent log of routing decisions and execution events
- **CLI entrypoint** (`researchswarm.py`) with `--run-prepare`, `--run-train`, `--route-only`, `--format` flags
- **Task classifier** (`researchswarm_agent.py`) with signal-based heuristics and segment-level classification
- **Built-in workflow executors**: `ResearchWorkflowExecutor`, `SummarizationExecutor`, `ReportExecutor`, `FileAnalysisExecutor`
- **Safety-first execution** — training scripts only run when explicit opt-in flags are passed
- **Forked base** from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) nanochat training stack
- GitHub Actions CI workflow
- Full test suite in `tests/`

### Changed
- Extended `program.md` with Digital Cognitive Labor research objectives
- Added `digital_cognitive_labor_program.md` as a broader instruction surface
- Updated `analysis.ipynb` with experiment analysis notebook

---

<sub>ResearchSwarm follows [Semantic Versioning](https://semver.org/). PATCH for bug fixes, MINOR for new features, MAJOR for breaking changes.</sub>
