.PHONY: install test smoke train-base train-film eval clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

smoke:
	python scripts/train_swinir.py --config configs/swinir_base.yaml --smoke

train-base:
	python scripts/train_swinir.py --config configs/swinir_base.yaml

train-film:
	python scripts/train_swinir.py --config configs/swinir_film.yaml

eval:
	python scripts/evaluate.py --checkpoint outputs/$(RUN)/checkpoints/best.pt --data-dir ../datasets

clean:
	rm -rf outputs/ wandb/ .pytest_cache/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
