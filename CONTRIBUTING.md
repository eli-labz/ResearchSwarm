# Contributing to ResearchSwarm

Thanks for your interest in contributing! This is a friendly, early-stage project.

## Quick Ways to Contribute

- ⭐ **Star the repo** if it helps you — it costs nothing and helps others find it
- 🐛 **Report bugs** via [GitHub Issues](https://github.com/eli-labz/ResearchSwarm/issues)
- 💡 **Suggest improvements** via [Discussions](https://github.com/eli-labz/ResearchSwarm/discussions)
- 🔧 **Submit a PR** (see guidelines below)

## Development Setup

```bash
git clone https://github.com/eli-labz/ResearchSwarm.git
cd ResearchSwarm
uv sync
```

Run the test suite:

```bash
uv run pytest tests/
```

## Contribution Guidelines

- **Keep `prepare.py` unmodified** — it is the fixed evaluation harness and must stay stable
- All other files are fair game
- Follow existing code style (PEP 8, type hints throughout)
- Add or update tests for new functionality in `tests/`
- Keep PRs focused on one logical change — easier to review and merge

## Good First Issues

| Area | Description |
|---|---|
| Platform ports | macOS (Metal/MLX), Windows, AMD — see forks table in README |
| New executors | Add a built-in executor for benchmark reporting or sweep summaries |
| Classifier improvements | Improve the cognitive labor task classifier accuracy |
| Memory queries | Better context injection from `AI-Memory/memory.db` |
| Visualization | Improve experiment plots in `analysis.ipynb` |

## PR Checklist

- [ ] Tests pass (`uv run pytest`)
- [ ] Code follows existing style
- [ ] `prepare.py` is unchanged
- [ ] PR description explains the change and why

## License

By contributing, you agree your work will be licensed under the [MIT License](LICENSE).
