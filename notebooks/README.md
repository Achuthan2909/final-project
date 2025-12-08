# Bayesian Temperature Scaling for Neural Network Calibration

## Project Structure

This project is organized into 9 sequential notebooks, each with a specific purpose. All notebooks use **PyStan** for Bayesian inference (as required).

### Notebook Workflow

```
01_setup_and_data.ipynb
    ↓
02_baseline_calibration.ipynb
    ↓
03_bayesian_temperature_scaling.ipynb
    ↓
04_uncertainty_quantification.ipynb
    ↓
05_advanced_bayesian_methods.ipynb
    ↓
06_calibration_metrics_uncertainty.ipynb
    ↓
07_active_learning_and_ood.ipynb
    ↓
08_results_and_visualization.ipynb
    ↓
09_summary_and_conclusions.ipynb
```

## Notebook Descriptions

### 01: Setup and Data Loading
**Purpose**: Initialize everything and extract logits

- Loads pre-trained ResNet56 model
- Loads CIFAR-10 dataset
- Extracts logits for validation and test sets
- Saves preprocessed data to `./data/processed/`

**Output Files**:
- `logits_val.npy`
- `labels_val.npy`
- `logits_test.npy`
- `labels_test.npy`

**Run Time**: ~2-3 minutes (model loading + logit extraction)

---

### 02: Baseline Calibration Methods
**Purpose**: Establish baseline for comparison

- Implements uncalibrated model
- Temperature scaling with L-BFGS (point estimate)
- Platt scaling
- Isotonic regression
- Compares all methods

**Output Files**:
- `baseline_results.npy` (saved to `./data/results/`)

**Key Result**: L-BFGS temperature estimate (used as prior center for Bayesian)

---

### 03: Bayesian Temperature Scaling (PyStan)
**Purpose**: Core Bayesian implementation

- Defines Stan model for temperature scaling
- Runs MCMC sampling (4 chains, 2000 samples, 1000 warmup)
- Analyzes posterior distribution
- Basic diagnostics (R-hat approximation, trace plots)

**Output Files**:
- `bayesian_posterior.npy` (saved to `./data/results/`)

**Key Result**: Posterior samples of temperature parameter

**Run Time**: ~10-15 minutes (MCMC sampling)

---

### 04: Uncertainty Quantification
**Purpose**: Demonstrate core value of Bayesian methods

**Part 1: Predictive Distributions**
- Uses full posterior to get uncertainty in predictions
- Shows which predictions are uncertain (not just point estimates)

**Part 2: Small Dataset Analysis**
- Compares Bayesian vs L-BFGS with varying validation sizes (100, 500, 1000, 5000)
- Shows how HDI width changes with data availability
- Demonstrates when uncertainty matters most

**Output Files**:
- `uncertainty_results.npy` (saved to `./data/results/`)

**Key Insight**: Bayesian provides uncertainty information that L-BFGS cannot

---

### 05: Advanced Bayesian Methods
**Purpose**: Extensions beyond single temperature

**Method 1: Per-Class Temperature Scaling**
- Estimates 10 temperatures (one per class)
- More flexible but more parameters

**Method 2: Prior Sensitivity Analysis**
- Tests different Gamma and Log-Normal priors
- Assesses robustness to prior choice

**Output Files**:
- `advanced_bayesian_results.npy` (saved to `./data/results/`)

**Run Time**: ~20-30 minutes (multiple MCMC runs)

---

### 06: Calibration Metrics with Uncertainty
**Purpose**: Quantify uncertainty in calibration quality itself

- Computes ECE and Brier score for each posterior sample
- Gets distributions of calibration metrics (not just point estimates)
- Enables risk assessment: "How confident are we that calibration improved?"

**Output Files**:
- `metric_uncertainty_results.npy` (saved to `./data/results/`)

**Key Insight**: Shows uncertainty in calibration quality, not just parameter uncertainty

---

### 07: Active Learning and OOD Analysis
**Purpose**: Practical applications of uncertainty quantification

**Part 1: Active Learning**
- Uses uncertainty to select which samples to label
- Shows faster uncertainty reduction with targeted sampling

**Part 2: Out-of-Distribution Analysis**
- Tests calibration on OOD data
- Assesses generalization of temperature scaling

**Output Files**:
- `application_results.npy` (saved to `./data/results/`)

**Run Time**: ~15-20 minutes (MCMC runs for active learning)

---

### 08: Results and Visualization
**Purpose**: Generate all figures for report

- Calibration curves (reliability diagrams)
- Posterior distribution plots
- Uncertainty vs sample size plots
- Summary tables

**Output Files**:
- All figures saved to `../latex/figures/`
  - `calibration_curves.pdf` / `.png`
  - `posterior_distribution.pdf` / `.png`
  - `uncertainty_vs_sample_size.pdf` / `.png`

---

### 09: Summary and Conclusions
**Purpose**: Comprehensive project summary

- Project overview
- Key findings
- Comparison of methods
- Limitations
- Future work
- Conclusions

---

## Quick Start

1. **Run notebooks in order** (01 → 09)
2. **First time**: Run Notebook 01 to extract logits
3. **Subsequent runs**: Can skip Notebook 01 if data already exists
4. **Note**: Notebooks 03, 05, and 07 involve MCMC sampling and take longer

## Data Flow

```
Notebook 01
    ↓ (saves to ./data/processed/)
Notebook 02
    ↓ (saves to ./data/results/baseline_results.npy)
Notebook 03
    ↓ (saves to ./data/results/bayesian_posterior.npy)
Notebooks 04-09
    ↓ (load previous results, save new results)
```

## Key Requirements

- **PyStan** (not PyMC) - all Bayesian inference uses PyStan
- PyTorch (for model loading and logits)
- NumPy, Matplotlib, scikit-learn
- CIFAR-10 dataset (auto-downloaded)

## File Structure

```
notebooks/
├── 01_setup_and_data.ipynb
├── 02_baseline_calibration.ipynb
├── 03_bayesian_temperature_scaling.ipynb
├── 04_uncertainty_quantification.ipynb
├── 05_advanced_bayesian_methods.ipynb
├── 06_calibration_metrics_uncertainty.ipynb
├── 07_active_learning_and_ood.ipynb
├── 08_results_and_visualization.ipynb
├── 09_summary_and_conclusions.ipynb
├── README.md (this file)
└── data/
    ├── processed/
    │   ├── logits_val.npy
    │   ├── labels_val.npy
    │   ├── logits_test.npy
    │   └── labels_test.npy
    └── results/
        ├── baseline_results.npy
        ├── bayesian_posterior.npy
        ├── uncertainty_results.npy
        ├── advanced_bayesian_results.npy
        ├── metric_uncertainty_results.npy
        └── application_results.npy
```

## Notes

- All notebooks use random seed 42 for reproducibility
- MCMC sampling uses 4 chains by default
- Posterior samples are saved for reuse in later notebooks
- Figures are saved in both PDF and PNG formats

## Troubleshooting

**Issue**: Stan model building is slow
- **Solution**: First build takes longer (compilation). Subsequent builds use cache.

**Issue**: MCMC sampling takes too long
- **Solution**: Reduce `num_samples` and `num_warmup` in sampling calls (but this reduces quality)

**Issue**: Out of memory
- **Solution**: Reduce number of posterior samples used in analysis (e.g., use `[::50]` instead of all samples)

---

**Project Goal**: Transform from "Bayesian estimation of one parameter" to "Comprehensive Bayesian uncertainty quantification for reliable machine learning predictions"



