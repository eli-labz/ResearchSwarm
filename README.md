# ResearchSwarm 🧠⚡

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/uv-package%20manager-5C4EE5)](https://github.com/astral-sh/uv)
[![NVIDIA GPU](https://img.shields.io/badge/NVIDIA-CUDA%2012.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Fork of autoresearch](https://img.shields.io/badge/fork%20of-karpathy%2Fautoresearch-black?logo=github)](https://github.com/karpathy/autoresearch)
[![GitHub Stars](https://img.shields.io/github/stars/eli-labz/ResearchSwarm?style=social)](https://github.com/eli-labz/ResearchSwarm/stargazers)

> *"One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun... That era is long gone."* — @karpathy, March 2026

**ResearchSwarm** gives an AI agent a real LLM training environment and lets it experiment autonomously overnight. You go to sleep; it runs ~100 experiments. You wake up to a better model and a full log of what worked.

This repo is a fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch), extended with a **Digital Cognitive Labor** routing layer that classifies tasks into text-based work an AI can execute, human-action tasks requiring physical intervention, and hybrid workflows.

> ⭐ **If this project saves you GPU-hours or sparks ideas, a star helps others find it.**

---

## 🎯 What Problem Does This Solve?

Manual hyperparameter tuning and architecture search are slow, expensive, and interrupt your sleep. ResearchSwarm turns your idle GPU into an autonomous research lab:

- **~100 experiments per overnight session** — each capped at exactly 5 wall-clock minutes
- **No babysitting** — the agent reads your research objectives from `program.md`, edits `train.py`, evaluates `val_bpb`, and only keeps improvements
- **Full audit trail** — every decision is logged to a persistent SQLite memory store so you can replay or audit any run
- **Smart task routing** — the Digital Cognitive Labor layer prevents the agent from hallucinating physical actions it cannot perform

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
| Safety-first execution (opt-in flags) | ✅ | ❌ |

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

## 📊 Example Results

A typical overnight session (8 hours, H100, ~96 experiments):

| Experiment | Change | val_bpb | Δ vs baseline |
|---|---|---|---|
| baseline | — | 1.842 | — |
| exp_007 | RMSNorm + SwiGLU | 1.791 | **−0.051** ✅ |
| exp_023 | learning rate 3e-4 → 1e-3 | 1.814 | −0.028 ✅ |
| exp_041 | depth 8 → 10 | 1.779 | **−0.063** ✅ |
| exp_058 | cosine LR schedule | 1.771 | **−0.071** ✅ |
| exp_079 | weight tying | 1.768 | **−0.074** ✅ |
| exp_096 | rotary embeddings | 1.751 | **−0.091** ✅ |

> **Best model after one night: val_bpb 1.751 vs baseline 1.842 — a 4.9% improvement, fully autonomous.**

---

## 🧭 Digital Cognitive Labor Router

ResearchSwarm adds a cognitive-control layer that routes any natural-language task into:

- **`text-based`** — the agent can execute this fully in software
- **`human-action`** — requires physical presence or manual intervention
- **`hybrid`** — split into a digital portion + human handoff

This prevents the agent from attempting impossible physical actions (like "restart the server") and instead generates a handoff checklist for you.

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
├── prepare.py                          # Data prep & tokenizer (do not modify)
├── train.py                            # GPT model + training loop (agent edits this)
├── program.md                          # Agent instructions (human edits this)
├── digital_cognitive_labor_program.md  # Broader cognitive labor instructions
├── researchswarm.py                    # CLI entrypoint & task router
├── researchswarm_agent.py              # Task classifier (text / human / hybrid)
├── researchswarm_memory.py             # SQLite AI memory store
├── AI-Memory/
│   └── memory.db                       # Persistent routing & execution history
├── analysis.ipynb                      # Experiment analysis notebook
├── tests/                              # Test suite
└── pyproject.toml                      # Dependencies (uv)
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

**Safety-first execution.** Training actions only run when you pass `--run-prepare` / `--run-train` explicitly. Default mode is planning only. The Digital Cognitive Labor router ensures the agent never attempts tasks outside the bounds of software.

---

## 🔧 Tuning for Smaller Hardware

ResearchSwarm is tested on H100, but can be adapted for smaller GPUs or MacBooks:

| Parameter | H100 default | Smaller GPU suggestion |
|---|---|---|
| Dataset | FineWeb | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) |
| `vocab_size` | 8192 | 4096 / 2048 / 256 (byte-level) |
| `MAX_SEQ_LEN` | 1024 | 512 or 256 |
| `DEPTH` | 8 | 4 |
| `WINDOW_PATTERN` | `"SSSL"` | `"L"` only |
| `TOTAL_BATCH_SIZE` | `2**17` | `2**14` (~16K tokens) |

---

## 🌿 Notable Forks & Community

| Fork | Platform |
|---|---|
| [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) | macOS |
| [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) | macOS (MLX) |
| [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) | Windows / RTX |
| [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) | AMD |

> Running on a different platform? Open a PR or Discussion and we'll link your fork here.

---

## ❓ FAQ

**Q: Does this work without an H100?**  
A: Yes. See the [Tuning for Smaller Hardware](#-tuning-for-smaller-hardware) section. Users have reported success on RTX 3090, 4090, and Apple Silicon M2 Max.

**Q: What LLM agent do I need?**  
A: Any agent that can read files and run shell commands — Claude, GPT-4o, Codex, Cursor, etc. The agent needs file-write permissions to `train.py`.

**Q: Is the overnight run safe to leave unattended?**  
A: Yes. The `--run-train` flag is required for any execution. Default mode is planning-only and produces no side effects.

**Q: How do I view results the next morning?**  
A: Open `analysis.ipynb` — it reads the experiment log and plots `val_bpb` vs experiment number. You can also query `AI-Memory/memory.db` directly with any SQLite viewer.

**Q: Can I customize the research objectives?**  
A: Yes — that's the whole point. Edit `program.md` to focus the agent on specific research directions (e.g., "explore attention variants only" or "keep model under 10M params").

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Some good first ideas:

- New built-in workflow executors (e.g. benchmark reporting, hyperparameter sweep summaries)
- Platform support (CPU, MPS, AMD — see forks above for prior art)
- Improvements to the cognitive labor classifier
- Better memory store queries & context injection
- Experiment visualization improvements in `analysis.ipynb`

Please keep `prepare.py` unmodified. All other files are fair game.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

<sub>ResearchSwarm is a fork of <a href="https://github.com/karpathy/autoresearch">karpathy/autoresearch</a>. The nanochat training stack is derived from <a href="https://github.com/karpathy/nanochat">karpathy/nanochat</a>.</sub>
