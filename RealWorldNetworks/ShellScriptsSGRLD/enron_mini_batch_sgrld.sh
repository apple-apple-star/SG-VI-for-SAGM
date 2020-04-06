#!/usr/bin/env bash

# FA
python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch.py -seed 0 -lrw 0.01 -lrpi 0.01 -n 'enron' -o '../Results/' -l 50 -m 50 -k 10 -ns 100 -b 5000 -s 5000 -tr 0.1
python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch.py -seed 0 -lrw 0.01 -lrpi 0.01 -n 'enron' -o '../Results/' -l 50 -m 50 -k 20 -ns 100 -b 5000 -s 5000 -tr 0.1
python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch.py -seed 0 -lrw 0.01 -lrpi 0.01 -n 'enron' -o '../Results/' -l 50 -m 50 -k 30 -ns 100 -b 5000 -s 5000 -tr 0.1
python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch.py -seed 0 -lrw 0.01 -lrpi 0.01 -n 'enron' -o '../Results/' -l 50 -m 50 -k 40 -ns 100 -b 5000 -s 5000 -tr 0.1
python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch.py -seed 0 -lrw 0.01 -lrpi 0.01 -n 'enron' -o '../Results/' -l 50 -m 50 -k 50 -ns 100 -b 5000 -s 5000 -tr 0.1


python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch.py -seed 0 -lrw 0.001 -lrpi 0.001 -n 'enron' -o '../Results/' -l 10 -m 10 -k 50 -ns 100 -b 50000 -s 10000 -tr 0.1
python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch.py -seed 0 -lrw 0.001 -lrpi 0.001 -n 'enron' -o '../Results/' -l 50 -m 50 -k 50 -ns 100 -b 40000 -s 10000 -tr 0.1
python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch.py -seed 0 -lrw 0.001 -lrpi 0.001 -n 'enron' -o '../Results/' -l 100 -m 100 -k 50 -ns 100 -b 30000 -s 10000 -tr 0.1
python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch.py -seed 0 -lrw 0.001 -lrpi 0.001 -n 'enron' -o '../Results/' -l 150 -m 150 -k 50 -ns 100 -b 20000 -s 10000 -tr 0.1
python -W ignore ../SGRLDMiniBatch/SGRLD_mini_batch.py -seed 0 -lrw 0.001 -lrpi 0.001 -n 'enron' -o '../Results/' -l 200 -m 200 -k 50 -ns 100 -b 10000 -s 10000 -tr 0.1
