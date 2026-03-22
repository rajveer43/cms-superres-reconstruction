PYTHON ?= . .venv/bin/activate && python

.PHONY: dataset-analysis train-final analyze-final presentation

dataset-analysis:
	$(PYTHON) analyze_dataset.py --data-dir datasets --out-dir reports/dataset_analysis_sample --batch-size 256 --max-events 2000

train-final:
	$(PYTHON) train_srgan.py --epochs 18 --batch-size 8 --stats-batch-size 16 --max-stats-batches 10 --max-train-batches 20 --max-val-batches 5 --lambda-l1 50 --lambda-physics 12 --output-dir outputs/final_run

analyze-final:
	$(PYTHON) analyze_results.py --run-dir outputs/final_run --data-dir datasets --batch-size 32 --max-batches 10

presentation:
	cp PRESENTATION.md reports/PRESENTATION.md
