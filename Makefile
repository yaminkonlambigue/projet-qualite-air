install:
	uv sync

collect:
	uv run python src/data_loader.py

notebook:
	uv run jupyter notebook

test:
	uv run pytest tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete