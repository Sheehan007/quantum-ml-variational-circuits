# Contributing

Contributions are welcome. Create a focused branch, add tests for behavioral
changes, and run the checks below before opening a pull request:

```bash
uv sync --extra dev --python 3.11
uv run ruff check .
uv run pytest
```

Experiment changes should keep random seeds explicit and distinguish simulated
noise from results obtained on quantum hardware.
