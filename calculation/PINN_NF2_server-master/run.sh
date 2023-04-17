#!/bin/bash
python scratch_extrapolate.py --config "configs/config_scratch_$1.json" &&
python series_extrapolate.py --config "configs/config_series_$1.json" &&
python series_evaluate.py --config "configs/config_series_$1.json"