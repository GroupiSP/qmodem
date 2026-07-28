.PHONY: battery-hnn battery-mcd battery-bnn battery-qavi battery-multiple-hnn battery-multiple-mcd battery-multiple-bnn battery-multiple-qavi generate-all battery-all clean-data clean-mlruns clean

battery-hnn:
	uv run scripts/battery/hnn_train.py && \
	uv run scripts/battery/hnn_test.py

battery-mcd:
	uv run scripts/battery/mcd_train.py && \
	uv run scripts/battery/mcd_test.py

battery-bnn:
	uv run scripts/battery/bnn_train.py && \
	uv run scripts/battery/bnn_test.py

battery-qavi:
	uv run scripts/battery/qavi_train.py && \
	uv run scripts/battery/qavi_test.py

battery-multiple-hnn:
	uv run scripts/battery_multiple_scenarios/hnn_train.py && \
	uv run scripts/battery_multiple_scenarios/hnn_test.py

battery-multiple-mcd:
	uv run scripts/battery_multiple_scenarios/mcd_train.py && \
	uv run scripts/battery_multiple_scenarios/mcd_test.py

battery-multiple-bnn:
	uv run scripts/battery_multiple_scenarios/bnn_train.py && \
	uv run scripts/battery_multiple_scenarios/bnn_test.py

battery-multiple-qavi:
	uv run scripts/battery_multiple_scenarios/qavi_train.py && \
	uv run scripts/battery_multiple_scenarios/qavi_test.py

generate-all:
	uv run scripts/battery/generate_discharge_histories.py && \
	uv run scripts/battery_multiple_scenarios/generate_discharge_histories.py

clean-data:
	find data/raw/battery data/raw/battery_multiple_scenarios data/processed -mindepth 1 ! -name '.gitkeep' -delete

clean-mlruns:
	uv run bash gc_mlflow.sh

battery-all: battery-hnn battery-mcd battery-bnn battery-qavi battery-multiple-hnn battery-multiple-mcd battery-multiple-bnn battery-multiple-qavi

clean: clean-data clean-mlruns
