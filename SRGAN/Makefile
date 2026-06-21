PYTHON ?= . ../.venv/bin/activate && python
DATA   ?= ../datasets

.PHONY: smoke-parquet smoke-hdf5 smoke-128 dataset-analysis train-final analyze-final help

## Smoke tests (run all three to verify the install)
smoke-parquet:
	$(PYTHON) train_srgan.py \
		--dataset-type parquet --dataset-path $(DATA) \
		--epochs 1 --batch-size 8 \
		--max-train-batches 5 --max-val-batches 2 \
		--output-dir outputs/smoke_parquet

smoke-hdf5:
	$(PYTHON) train_srgan.py \
		--dataset-type hdf5 --dataset-path $(DATA)/calochallenge_dataset2 \
		--epochs 1 --batch-size 8 \
		--max-train-batches 5 --max-val-batches 2 \
		--output-dir outputs/smoke_hdf5

smoke-128:
	$(PYTHON) train_srgan.py \
		--dataset-type parquet --dataset-path $(DATA) \
		--hr-size 128 128 --scale-factor 2.0 \
		--epochs 1 --batch-size 8 --max-train-batches 5 \
		--output-dir outputs/smoke_128x128

## Analysis helpers
dataset-analysis:
	$(PYTHON) analyze_dataset.py \
		--data-dir $(DATA) \
		--out-dir reports/dataset_analysis_sample \
		--batch-size 256 --max-events 2000

train-final:
	$(PYTHON) train_srgan.py \
		--dataset-type parquet --dataset-path $(DATA) \
		--epochs 18 --batch-size 8 \
		--stats-batch-size 16 --max-stats-batches 10 \
		--max-train-batches 20 --max-val-batches 5 \
		--lambda-l1 50 --lambda-physics 12 \
		--output-dir outputs/final_run

analyze-final:
	$(PYTHON) analyze_results.py \
		--run-dir outputs/final_run \
		--data-dir $(DATA) \
		--batch-size 32 --max-batches 10

help:
	@echo "Targets:"
	@echo "  smoke-parquet   1-epoch sanity check on parquet dataset"
	@echo "  smoke-hdf5      1-epoch sanity check on CaloChallenge HDF5 dataset"
	@echo "  smoke-128       1-epoch sanity check with 128x128 HR output"
	@echo "  dataset-analysis  Run EDA on parquet files"
	@echo "  train-final     Full training run (parquet)"
	@echo "  analyze-final   Generate plots and physics metrics for final_run"
	@echo ""
	@echo "Override data dir: make smoke-parquet DATA=/path/to/datasets"
