.PHONY: generate-all clean-data clean-mlruns clean

generate-all:
	uv run scripts/battery/generate_discharge_histories.py && \
	uv run scripts/battery_multiple_scenarios/generate_discharge_histories.py

clean-data:
	find data/raw/battery data/raw/battery_multiple_scenarios data/processed -mindepth 1 ! -name '.gitkeep' -delete

clean-mlruns:
	uv run bash gc_mlflow.sh

clean: clean-data clean-mlruns
