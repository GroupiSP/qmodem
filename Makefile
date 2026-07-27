.PHONY: clean clean-data clean-mlruns

clean-data:
	find data/raw/battery data/raw/multiple_scenarios data/processed -mindepth 1 ! -name '.gitkeep' -delete

clean-mlruns:
	bash gc_mlflow.sh

clean: clean-data clean-mlruns
