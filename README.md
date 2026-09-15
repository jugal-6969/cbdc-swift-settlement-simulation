# CBDC vs SWIFT: Cross-Border Settlement-Time Simulation

This repository contains the simulation and analysis code for the MSc dissertation
*Evaluating CBDC-Based Cross-Border Payment Systems as a Mechanism for Improving
Settlement-Time Predictability* (module MSO4992, Middlesex University).

The study compares a SWIFT-style correspondent settlement rail against a CBDC
settlement rail (modelled on Project mBridge) across four corridors: US-India,
US-China, UAE-India and UAE-Pakistan. It measures how settlement time, and above all
the **variance** of settlement time, differs between the two rails, and identifies the
conditions (number of intermediaries and destination income level) under which the
predictability gap is largest.

## What this repository contains

| File | Description |
|------|-------------|
| `simulation.py` | The full simulation and statistical analysis. Running it regenerates every result and figure used in Chapter 5. |
| `results.json` | The numerical results produced by the simulation (effect sizes, test statistics, the threshold grid, corridor outcomes). |
| `fig_5_1_distributions.png` | Figure 5.1: settlement-time distributions for the two rails. |
| `fig_5_2_intermediaries.png` | Figure 5.2: SWIFT settlement time and spread by intermediary count. |
| `fig_5_3_income.png` | Figure 5.3: SWIFT settlement time by destination income band. |
| `fig_5_4_threshold_heatmap.png` | Figure 5.4: the predictability gap across the full grid. |
| `requirements.txt` | The Python libraries needed to run the code. |

## Requirements

- Python 3.10 or newer
- The libraries listed in `requirements.txt`

## How to run

1. Install the required libraries:

   ```
   pip install -r requirements.txt
   ```

2. Run the simulation:

   ```
   python simulation.py
   ```

This regenerates `results.json`, the four figures, and a large CSV of the raw
simulated settlement times (`simulated_settlement_times.csv`). The run takes only a
few seconds.

## Reproducibility

The simulation uses a fixed random seed (`SEED = 42`), so the results are fully
reproducible. Running the script on a clean environment reproduces `results.json`
exactly and regenerates the four figures identically. This is what makes the reported
statistics verifiable rather than something to be taken on trust.

The results were produced and verified with the following versions:
numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, scikit-posthocs 0.17.0, matplotlib 3.10.8.
Because `numpy.random.default_rng` produces a stable stream across versions, the
results reproduce on other reasonably recent versions of these libraries as well.

## Data sources

The simulation is calibrated from published, publicly available sources:

- World Bank, Remittance Prices Worldwide (RPW) dataset, 2011-2025.
- SWIFT gpi settlement-time statistics (Nilsson et al., 2022, BIS/CPMI).
- BIS and HKMA publications on Project mBridge.

No raw World Bank data file is included in this repository; the baseline analysis in
Chapter 4 uses the public RPW dataset linked above, and the simulation is calibrated
from the published figures rather than from that file.

## Author

Jugal Bhagat, MSc Financial Technology, Middlesex University (2026).

## License

Released under the MIT License. See `LICENSE`.
