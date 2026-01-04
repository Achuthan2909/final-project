# Bayesian Temperature Scaling for Neural Network Calibration

**Final Project for MA 578: Bayesian Statistics**

[![Project Report](latex/FinalReport.pdf)](latex/FinalReport.pdf)

## Overview

This project implements **Bayesian temperature scaling** for calibrating neural network predictions, providing uncertainty quantification beyond standard point estimation methods. We demonstrate that Bayesian methods offer crucial advantages for safety-critical applications by quantifying uncertainty in both parameter estimates and calibration quality.

### Key Contributions

- **Uncertainty Quantification**: Bayesian MCMC provides 95% credible intervals, showing how confidence scales with data availability
- **Calibration Improvement**: Reduced Expected Calibration Error (ECE) from 0.0386 to 0.0091 (76.4% improvement) while maintaining 94.4% accuracy
- **Robustness Analysis**: Prior sensitivity analysis demonstrates robustness with sufficient data
- **Practical Applications**: Demonstrates active learning and out-of-distribution detection using uncertainty estimates

## Problem Statement

Modern neural networks achieve high accuracy but often exhibit poor calibration: predicted probabilities do not match actual correctness rates. Standard temperature scaling uses L-BFGS optimization to find a single temperature parameter, providing only a point estimate with no uncertainty information. **Bayesian temperature scaling** treats the temperature as a random variable and estimates its full posterior distribution using MCMC, enabling uncertainty quantification essential for:

- Medical diagnosis systems
- Autonomous vehicles
- Financial risk assessment
- Limited data scenarios

## Dataset and Model

- **Dataset**: CIFAR-10 (60,000 images, 10 classes)
- **Model**: Pre-trained ResNet56 (94.4% test accuracy)
- **Data Split**: 
  - Training: 50,000 images
  - Validation: 5,000 images (for calibration)
  - Test: 5,000 images (for evaluation)

## Methods

### Temperature Scaling Framework

Given logits $\mathbf{z} \in \mathbb{R}^K$, temperature scaling applies scalar $T > 0$:

$$p_k = \text{softmax}(\mathbf{z} / T)_k = \frac{\exp(z_k / T)}{\sum_{j=1}^K \exp(z_j / T)}$$

### Bayesian Approach

- **Likelihood**: $y_i \mid T, \mathbf{z}_i \sim \text{Categorical}(\text{softmax}(\mathbf{z}_i / T))$
- **Prior**: $T \sim \text{Gamma}(\alpha=4, \beta=4/T_0)$ where $T_0$ is the L-BFGS estimate
- **Inference**: MCMC via NUTS in PyStan (4 chains, 2,000 samples, 1,000 warmup)
- **Estimator**: Posterior mean $\hat{T} = \mathbb{E}[T \mid \mathbf{y}, \mathbf{Z}]$

## Key Results

### Calibration Performance

| Method | Temperature | ECE | Brier Score | Uncertainty |
|--------|------------|-----|-------------|-------------|
| Uncalibrated | 1.000 | 0.0386 | 0.0943 | N/A |
| L-BFGS | 1.726 | 0.0094 | 0.0860 | N/A |
| **Bayesian** | **1.728** | **0.0091** | **0.0860** | **[0.0061, 0.0134]** |

**76.4% improvement in ECE** while maintaining accuracy.

### Uncertainty Quantification

- **With 5,000 samples**: Posterior mean T = 1.728, 95% HDI [1.664, 1.792] (width: 0.128)
- **With 100 samples**: 95% HDI [0.55, 1.95] (width: 1.000)
- **87.2% reduction** in uncertainty width as sample size increases

The Bayesian approach appropriately reflects uncertainty when data is limited, warning against overconfident deployment decisions.

### Robustness

- Posterior mean (1.728) matches L-BFGS estimate (1.726)
- With sufficient data (n=5000), different priors yield essentially identical posteriors (range < 0.01)
- Prior sensitivity analysis confirms robustness with adequate sample sizes

## Project Structure

```
final-project/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── Project-Proposal.pdf         # Original project proposal
├── latex/
│   ├── FinalReport.tex          # LaTeX source
│   ├── FinalReport.pdf          # Final report (see link above)
│   └── figures/                 # All visualization figures
├── notebooks/
│   ├── README.md                # Detailed notebook documentation
│   ├── 01_setup_and_data.ipynb
│   ├── 02_baseline_calibration.ipynb
│   ├── 03_bayesian_temperature_scaling.ipynb  # Core Bayesian implementation
│   ├── 04_uncertainty_quantification.ipynb
│   ├── 05_advanced_bayesian_methods.ipynb
│   ├── 06_calibration_metrics_uncertainty.ipynb
│   ├── 07_active_learning_and_ood.ipynb
│   ├── 08_results_and_visualization.ipynb
│   └── 09_summary_and_conclusions.ipynb
├── scripts/
│   ├── install.sh               # Installation script
│   └── compile_latex.sh         # LaTeX compilation script
└── docs/
    └── INSTALLATION.md          # PyStan installation guide (ARM64 Mac)
```

## Quick Start

### Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install PyStan** (required for Bayesian inference):
   - Standard installation: `pip install pystan`
   - For ARM64 Mac (M1/M2/M3): See `docs/INSTALLATION.md` for detailed instructions

3. **Run notebooks sequentially** (01 → 09):
   ```bash
   cd notebooks
   jupyter notebook
   ```

### Running the Analysis

1. **Notebook 01**: Load pre-trained model and extract logits (~2-3 minutes)
2. **Notebook 02**: Baseline calibration methods (L-BFGS, Platt scaling, etc.)
3. **Notebook 03**: Core Bayesian temperature scaling with MCMC (~10-15 minutes)
4. **Notebooks 04-09**: Extended analysis, uncertainty quantification, and visualization

**Note**: Notebooks 03, 05, and 07 involve MCMC sampling and take longer to run.

## Key Findings

1. **Uncertainty Quantification**: Bayesian methods provide credible intervals that appropriately scale with data availability, offering crucial insights for deployment decisions
2. **Calibration Improvement**: Temperature scaling significantly improves calibration (76.4% ECE reduction) without sacrificing accuracy
3. **Robustness**: Results are robust to prior specification when sufficient data is available
4. **Model Validation**: Posterior predictive checks confirm appropriate model specification

## Computational Considerations

- **MCMC Sampling**: ~10-15 seconds (vs <1 second for L-BFGS)
- **Overhead**: Negligible for one-time calibration after training
- **Trade-off**: Small computational cost for substantial gains in uncertainty quantification

## Limitations

- Computational cost higher than point estimation (but minimal for one-time calibration)
- Assumes validation set is representative of test distribution
- CIFAR-10 is relatively balanced; results may differ with class imbalance

## Future Work

- Per-class temperature scaling with hierarchical priors
- Application to other calibration methods (Platt scaling, isotonic regression)
- Extension to regression tasks
- Real-time calibration with approximate Bayesian inference

## Report

See the complete project report: **[FinalReport.pdf](latex/FinalReport.pdf)**

The report includes:
- Detailed methodology and theoretical background
- Complete results and analysis
- All visualizations and figures
- Prior sensitivity analysis
- Posterior predictive checks
- Stan model code

## Citation

If you use this work, please cite:

```bibtex
@misc{bayesian_temperature_scaling_2025,
  title={Bayesian Temperature Scaling for Neural Network Calibration},
  author={Achuthan},
  year={2025},
  note={MA 578: Bayesian Statistics Final Project}
}
```

## References

- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*.
- Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). Chapman and Hall/CRC.
- Stan Development Team. (2024). *Stan Modeling Language Users Guide*.

## License

This project is for educational purposes as part of MA 578: Bayesian Statistics coursework.

## Contact

For questions or issues, please open an issue on GitHub.

---

**Project Goal**: Transform from "Bayesian estimation of one parameter" to "Comprehensive Bayesian uncertainty quantification for reliable machine learning predictions"

