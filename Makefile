.PHONY: bench test lint

bench:
	uv run python bench/overhead.py --write

test:
	uv run pytest -q

lint:
	uv run ruff check .
