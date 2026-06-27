# ResearchSwarm 🧠⚡

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/uv-package%20manager-5C4EE5)](https://github.com/astral-sh/uv)
[![NVIDIA GPU](https://img.shields.io/badge/NVIDIA-CUDA%2012.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Fork of autoresearch](https://img.shields.io/badge/fork%20of-karpathy%2Fautoresearch-black?logo=github)](https://github.com/karpathy/autoresearch)

> *"One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun... That era is long gone."* — @karpathy, March 2026

**ResearchSwarm** gives an AI agent a real LLM training environment and lets it experiment autonomously overnight. You go to sleep; it runs ~100 experiments. You wake up to a better model and a full log of what worked.

This repo is a fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch), extended with a **Digital Cognitive Labor** routing layer that classifies tasks into text-based work an AI can execute, human-action tasks requiring physical intervention, and hybrid workflows.

---

## ✨ What Makes ResearchSwarm Different

| Feature | ResearchSwarm | vanilla autoresearch |
|---|---|---|
| Autonomous overnight LLM training | ✅ | ✅ |
| Digital Cognitive Labor router | ✅ | ❌ |
| Task classifier (text / human / hybrid) | ✅ | ❌ |
| AI memory store (SQLite) | ✅ | ❌ |
| CLI entrypoint with safety flags | ✅ | ❌ |
| Built-in workflow executors | ✅ | ❌ |

---

## 🚀 Quick Start

### Requirements

- Single NVIDIA GPU (tested on H100)
- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) package manager

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repo
git clone https://github.com/eli-labz/ResearchSwarm.git
cd ResearchSwarm

# 3. Install dependencies
uv sync

# 4. Download data & train tokenizer (one-time, ~2 min)
uv run prepare.py

# 5. Run a single training experiment (~5 min)
uv run train.py
```

If all five steps complete successfully, your setup is working. Now go autonomous.

---

## 🤖 Autonomous Research Mode

Point your AI coding agent (Claude, Codex, etc.) at this repo — with file permissions enabled so it can edit `train.py` — then prompt it:

```
Have a look at program.md and let's kick off a new experiment! Let's do the setup first.
```

The agent will:
1. Read `program.md` for instructions
2. Edit `train.py` (architecture, hyperparameters, optimizer, etc.)
3. Run a 5-minute training experiment
4. Evaluate `val_bpb` (validation bits-per-byte) — lower is better
5. Keep the change if it improved, discard if not
6. Repeat ~100 times while you sleep

**You wake up to a better model and a full experiment log.**

> The key insight: you are not touching Python files directly. You are *programming the program* by editing `program.md` to give the agent better instructions.

---

## 🧭 Digital Cognitive Labor Router

ResearchSwarm adds a cognitive-control layer that routes any natural-language task into:

- **`text-based`** — the agent can execute this fully in software
- **`human-action`** — requires physical presence or manual intervention
- **`hybrid`** — split into a digital portion + human handoff

### CLI Examples

```bash
# Overnight training run (planning mode — safe by default)
uv run researchswarm "Prepare the data and run a baseline training experiment overnight"

# Actually execute training (opt-in flags required)
uv run researchswarm --run-prepare --run-train "Prepare the data and run a baseline training experiment overnight"

# Text-based tasks (execute immediately)
uv run researchswarm "Draft a postmortem from yesterday's run logs"
uv run researchswarm --file README.md "Analyze this file"

# Human-action tasks (router produces a handoff checklist instead)
uv run researchswarm "Go to the server room and reseat the GPU power cable"
```

### Task Classification Example

```bash
uv run researchswarm_agent "Summarize the training logs and then physically restart the server"
```

```json
{
  "domain": "hybrid",
  "confidence": 0.75,
  "digital_segments": ["Summarize the training logs"],
  "human_segments": ["physically restart the server"],
  "recommended_action": "Complete the digital portion now, isolate the physical/manual portion, and create an explicit handoff boundary."
}
```

---

## 🗂️ Project Structure

```
ResearchSwarm/
├── prepare.py                         # Data prep & tokenizer (do not modify)
├── train.py                           # GPT model + training loop (agent edits this)
├── program.md                         # Agent instructions (human edits this)
├── digital_cognitive_labor_program.md # Broader cognitive labor instructions
├── researchswarm.py                   # CLI entrypoint & task router
├── researchswarm_agent.py             # Task classifier (text / human / hybrid)
├── researchswarm_memory.py            # SQLite AI memory store
├── AI-Memory/
│   └── memory.db                      # Persistent routing & execution history
├── analysis.ipynb                     # Experiment analysis notebook
├── tests/                             # Test suite
└── pyproject.toml                     # Dependencies (uv)
```

**The three files that matter for training:**

| File | Who edits it | What it does |
|---|---|---|
| `prepare.py` | Nobody | Fixed data prep & utilities |
| `train.py` | The AI agent | Full GPT model, optimizer, training loop |
| `program.md` | You | Instructions & research objectives for the agent |

---

## ⚙️ Design Philosophy

**Fixed 5-minute time budget.** Every experiment runs for exactly 5 wall-clock minutes (~12 experiments/hour, ~100 overnight). All experiments are directly comparable regardless of what the agent changes (model size, batch size, architecture), and results are optimized for *your specific hardware*.

**Single metric.** `val_bpb` (validation bits-per-byte) — lower is better. Vocab-size-independent so architectural changes are fairly compared.

**One file to modify.** The agent only touches `train.py`. Everything is in scope: architecture, hyperparameters, optimizer choice, attention patterns. Diffs stay reviewable.

**Memory-grounded.** Routing decisions and execution events are logged to `AI-Memory/memory.db`. Recent context is surfaced back into each new task so the agent stays grounded in prior decisions.

**Safety-first execution.** Training actions only run when you pass `--run-prepare` / `--run-train` explicitly. Default mode is planning only.

---

## 🔧 Tuning for Smaller Hardware

ResearchSwarm is tested on H100, but can be adapted for smaller GPUs or Macbooks:

- **Dataset**: Use [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) for narrower-scope data that works at smaller scale
- **`vocab_size`**: Lower from 8192 to 4096, 2048, 1024, or byte-level (256)
- **`MAX_SEQ_LEN`** in `prepare.py`: Reduce to 512 or 256
- **`DEPTH`** in `train.py`: Default is 8; try 4 for smaller models
- **`WINDOW_PATTERN`**: Use `"L"` only — `"SSSL"` banded attention may be slow on non-H100
- **`TOTAL_BATCH_SIZE`**: Lower to powers of 2, e.g. `2**14` (~16K tokens)

---

## 🌿 Notable Forks

| Fork | Platform |
|---|---|
| [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) | macOS |
| [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) | macOS (MLX) |
| [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) | Windows / RTX |
| [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) | AMD |

> Running on a different platform? Open a PR or Discussion and we'll link your fork here.

---

## 🤝 Contributing

Contributions welcome! Some ideas:

- New built-in workflow executors (e.g. benchmark reporting, hyperparameter sweep summaries)
- Platform support (CPU, MPS, AMD — see forks above for prior art)
- Improvements to the cognitive labor classifier
- Better memory store queries & context injection

Please keep `prepare.py` unmodified. All other files are fair game.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

<sub>ResearchSwarm is a fork of <a href="https://github.com/karpathy/autoresearch">karpathy/autoresearch</a>. The nanochat training stack is derived from <a href="https://github.com/karpathy/nanochat">karpathy/nanochat</a>.</sub>
