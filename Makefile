.PHONY: all install data pipeline excel verify-excel test lint format dashboard screenshots clean

all: install pipeline lint test

install:
	uv sync

data:
	uv run field-ops generate

# generate -> clean/classify -> metrics -> Excel report -> recalculate it in LibreOffice
pipeline:
	uv run field-ops all

excel:
	uv run field-ops excel

# fails on any formula error; set FIELD_OPS_REQUIRE_LIBREOFFICE=1 to fail if LibreOffice is missing
verify-excel:
	uv run field-ops verify-excel

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

dashboard:
	uv run streamlit run app/dashboard.py

# README screenshots of the running dashboard (needs `make all` first). Not part of `make all`.
screenshots:
	uv run playwright install chromium
	uv run python scripts/screenshots.py

clean:
	rm -rf data/raw/*.csv data/truth output/*.csv output/*.xlsx output/*.md output/metrics output/data_quality
