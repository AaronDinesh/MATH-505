#!/bin/bash

start_runs=41; for j in {30..190..10}; do for i in {start_runs..start_runs+4}; do sbatch tsqr.run $i $j; done start_runs=start_runs+4; done	
