# SG-VI-for-SAGM
Stochastic Gradient Variational Inference(SG-VI) for S-AGM
## Prerequisites

Python 2.7.10 or above.

Required libraries for python: numpy-1.14.5, tensorflow-1.10.1, scipy-1.2.2. tensorflow-probability-0.3.0

## Synthetic Networks

### To reproduce the results with SGRLD
```
cd SG-VI-for-SAGM/SyntheticGraphs/ShellScriptsSGRLD/
chmod 777 ./*.sh
./synthetic_AGM.sh
./synthetic_aMMSB.sh
```

### To reproduce the results with SG-VI
```
cd SG-VI-for-SAGM/SyntheticGraphs/ShellScriptsSVI/
chmod 777 ./*.sh
./synthetic_AGM.sh
./synthetic_aMMSB.sh
./synthetic_reimann_AGM.sh
./synthetic_reimann_aMMSB.sh
```

### The folder for plots in paper
```
cd SG-VI-for-SAGM/SyntheticGraphs/Plots/
```
## Real World Networks

### To reproduce the results with SGRLD
```
cd SG-VI-for-SAGM/RealWorldNetworks/ShellScriptsSGRLD/
chmod 777 ./*.sh
./caHepPh.sh
./enron.sh
./FA.sh
./Reuters.sh
```

### To reproduce the results with SG-VI
```
cd SG-VI-for-SAGM/RealWorldNetworks/ShellScriptsSVI/
chmod 777 ./*.sh
./caHepPh.sh
./enron.sh
./FA.sh
./Reuters.sh
```

### The folder for plots in paper
```
cd SG-VI-for-SAGM/RealWorldNetworks/Plots/
```
