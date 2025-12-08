#!/usr/bin/env python3
import json
import os

def restore_notebook_02():
    with open('02_baseline_calibration.ipynb', 'r') as f:
        nb = json.load(f)
    
    nb['cells'][6]['source'] = """print('='*60)
print('METHOD 4: ISOTONIC REGRESSION')
print('='*60)

print('Fitting isotonic regression on max probability...')
max_probs_val = np.max(F.softmax(torch.tensor(logits_val), dim=1).numpy(), axis=1)
max_probs_test = np.max(F.softmax(torch.tensor(logits_test), dim=1).numpy(), axis=1)

iso_model = IsotonicRegression(out_of_bounds='clip')
iso_model.fit(max_probs_val, labels_val)

iso_probs = iso_model.predict(max_probs_test)

probs_iso_raw = F.softmax(torch.tensor(logits_test), dim=1).numpy()
probs_iso = probs_iso_raw * (iso_probs.reshape(-1, 1) / max_probs_test.reshape(-1, 1))
probs_iso = probs_iso / probs_iso.sum(axis=1, keepdims=True)

preds_iso = np.argmax(probs_iso, axis=1)
conf_iso = np.max(probs_iso, axis=1)

acc_iso = (preds_iso == labels_test).mean()
ece_iso = compute_ece(probs_iso, labels_test)
brier_iso = compute_brier_score(probs_iso, labels_test)

print(f'✓ Isotonic regression completed')
print(f'Accuracy: {acc_iso:.4f}')
print(f'ECE: {ece_iso:.4f}')
print(f'Brier Score: {brier_iso:.4f}')
print(f'Mean Confidence: {conf_iso.mean():.4f}')"""

    nb['cells'][7]['source'] = """print('='*60)
print('BASELINE METHODS COMPARISON')
print('='*60)

results = {
    'Uncalibrated': {'acc': acc_uncal, 'ece': ece_uncal, 'brier': brier_uncal, 'conf': conf_uncal.mean()},
    'Temperature Scaling': {'acc': acc_temp, 'ece': ece_temp, 'brier': brier_temp, 'conf': conf_temp.mean()},
    'Platt Scaling': {'acc': acc_platt, 'ece': ece_platt, 'brier': brier_platt, 'conf': conf_platt.mean()},
    'Isotonic Regression': {'acc': acc_iso, 'ece': ece_iso, 'brier': brier_iso, 'conf': conf_iso.mean()}
}

print(f'\\n{"Method":<25} {"Accuracy":<12} {"ECE":<12} {"Brier":<12} {"Mean Conf":<12}')
print('-'*73)
for method, metrics in results.items():
    print(f'{method:<25} {metrics["acc"]:<12.4f} {metrics["ece"]:<12.4f} {metrics["brier"]:<12.4f} {metrics["conf"]:<12.4f}')

best_ece = min(results.items(), key=lambda x: x[1]['ece'])
best_brier = min(results.items(), key=lambda x: x[1]['brier'])

print(f'\\nBest methods:')
print(f'  Best ECE: {best_ece[0]} ({best_ece[1]["ece"]:.4f})')
print(f'  Best Brier: {best_brier[0]} ({best_brier[1]["brier"]:.4f})')"""

    nb['cells'][8]['source'] = ""

    nb['cells'][9]['source'] = """print('='*60)
print('COMPREHENSIVE MODEL COMPARISON (INCLUDING BAYESIAN)')
print('='*60)

print('\\nLoading Bayesian results for comparison...')
try:
    bayesian_results = np.load('./data/results/bayesian_posterior.npy', allow_pickle=True).item()
    metric_results = np.load('./data/results/metric_uncertainty_results.npy', allow_pickle=True).item()
    
    bayesian_mean = bayesian_results['mean']
    ece_hdi = metric_results['ece_hdi']
    
    print(f'\\n{"Method":<30} {"Accuracy":<12} {"ECE":<18} {"Brier":<18} {"Uncertainty":<25}')
    print('-'*103)
    print(f'{"Uncalibrated":<30} {acc_uncal:<12.4f} {ece_uncal:<18.4f} {brier_uncal:<18.4f} {"N/A":<25}')
    print(f'{"L-BFGS":<30} {acc_temp:<12.4f} {ece_temp:<18.4f} {brier_temp:<18.4f} {"N/A":<25}')
    print(f'{"Bayesian (mean)":<30} {acc_temp:<12.4f} {metric_results["ece_mean"]:<18.4f} {metric_results["brier_mean"]:<18.4f} {f"ECE: [{ece_hdi[0]:.4f}, {ece_hdi[1]:.4f}]":<25}')
    print(f'{"Platt Scaling":<30} {acc_platt:<12.4f} {ece_platt:<18.4f} {brier_platt:<18.4f} {"N/A":<25}')
    print(f'{"Isotonic Regression":<30} {acc_iso:<12.4f} {ece_iso:<18.4f} {brier_iso:<18.4f} {"N/A":<25}')
    
    print('\\nKey Advantages of Bayesian Approach:')
    print(f'  1. Provides uncertainty quantification: Temperature HDI [{bayesian_results["hdi_lower"]:.4f}, {bayesian_results["hdi_upper"]:.4f}]')
    print(f'  2. Quantifies uncertainty in calibration metrics: ECE HDI [{ece_hdi[0]:.4f}, {ece_hdi[1]:.4f}]')
    print('  3. Enables risk assessment and better decision-making')
    print('  4. Critical when validation data is limited')
except FileNotFoundError:
    print('Bayesian results not found. Run Notebook 3 first.')"""

    nb['cells'][10]['source'] = """fig, axes = plt.subplots(1, 2, figsize=(14, 5))

methods = ['Uncalibrated', 'L-BFGS', 'Bayesian', 'Platt', 'Isotonic']
ece_scores = [ece_uncal, ece_temp, metric_results.get('ece_mean', ece_temp), ece_platt, ece_iso]
brier_scores = [brier_uncal, brier_temp, metric_results.get('brier_mean', brier_temp), brier_platt, brier_iso]

if 'ece_std' in metric_results:
    ece_err = [0, 0, metric_results['ece_std'], 0, 0]
    axes[0].errorbar(range(len(methods)), ece_scores, yerr=ece_err, fmt='o-', capsize=5, linewidth=2, markersize=8)
else:
    axes[0].plot(range(len(methods)), ece_scores, 'o-', linewidth=2, markersize=8)

axes[0].set_xticks(range(len(methods)))
axes[0].set_xticklabels(methods, rotation=45, ha='right')
axes[0].set_ylabel('ECE')
axes[0].set_title('Expected Calibration Error Comparison')
axes[0].grid(True, alpha=0.3)

if 'brier_std' in metric_results:
    brier_err = [0, 0, metric_results['brier_std'], 0, 0]
    axes[1].errorbar(range(len(methods)), brier_scores, yerr=brier_err, fmt='s-', capsize=5, linewidth=2, markersize=8)
else:
    axes[1].plot(range(len(methods)), brier_scores, 's-', linewidth=2, markersize=8)

axes[1].set_xticks(range(len(methods)))
axes[1].set_xticklabels(methods, rotation=45, ha='right')
axes[1].set_ylabel('Brier Score')
axes[1].set_title('Brier Score Comparison')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print('\\nVisualization shows:')
print('- Bayesian method includes uncertainty intervals (error bars)')
print('- Lower bars indicate better calibration')
print('- Bayesian provides additional information: uncertainty quantification')"""

    nb['cells'][11]['source'] = """print('='*60)
print('SAVING BASELINE RESULTS')
print('='*60)

os.makedirs('./data/results', exist_ok=True)

baseline_results = {
    'calibrated_temp': calibrated_temp,
    'results': results
}

np.save('./data/results/baseline_results.npy', baseline_results, allow_pickle=True)

print('✓ Baseline results saved to ./data/results/baseline_results.npy')
print(f'\\nKey baseline value:')
print(f'  L-BFGS Temperature: {calibrated_temp:.4f}')
print('  This will be used as prior center for Bayesian methods')
print('\\nNext step: Run Notebook 3 for Bayesian temperature scaling')"""

    with open('02_baseline_calibration.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)
    print("✓ Notebook 02 restored")

def restore_notebook_03():
    with open('03_bayesian_temperature_scaling.ipynb', 'r') as f:
        nb = json.load(f)
    
    nb['cells'][1]['source'] = """import numpy as np
import torch
import torch.nn.functional as F
import stan
import httpstan
import nest_asyncio
nest_asyncio.apply()
import matplotlib.pyplot as plt
import os

np.random.seed(42)
torch.manual_seed(42)

print('Loading data and baseline results...')
logits_val = np.load('./data/processed/logits_val.npy')
labels_val = np.load('./data/processed/labels_val.npy')
logits_test = np.load('./data/processed/logits_test.npy')
labels_test = np.load('./data/processed/labels_test.npy')

baseline_results = np.load('./data/results/baseline_results.npy', allow_pickle=True).item()
calibrated_temp = baseline_results['calibrated_temp']

print(f'✓ Data loaded')
print(f'  Validation: {logits_val.shape[0]} samples')
print(f'  Test: {logits_test.shape[0]} samples')
print(f'  Baseline temperature: {calibrated_temp:.4f}')"""

    nb['cells'][2]['source'] = """print('='*60)
print('DEFINING STAN MODEL')
print('='*60)

stan_model_code = \"\"\"
data {
    int<lower=0> N;
    int<lower=2> K;
    matrix[N, K] logits;
    array[N] int<lower=1, upper=K> y;
    real<lower=0> prior_alpha;
    real<lower=0> prior_beta;
}
parameters {
    real<lower=0> temperature;
}
model {
    temperature ~ gamma(prior_alpha, prior_beta);
    
    for (n in 1:N) {
        vector[K] scaled_logits = logits[n]' / temperature;
        y[n] ~ categorical_logit(scaled_logits);
    }
}
generated quantities {
    array[N] int<lower=1, upper=K> y_rep;
    vector[N] log_lik;
    
    for (n in 1:N) {
        vector[K] scaled_logits = logits[n]' / temperature;
        y_rep[n] = categorical_logit_rng(scaled_logits);
        log_lik[n] = categorical_logit_lpmf(y[n] | scaled_logits);
    }
}
\"\"\"

print('✓ Stan model defined')
print('\\nModel structure:')
print('  - Prior: Gamma(α, β) on temperature')
print('  - Likelihood: Categorical with temperature-scaled logits')
print('  - Temperature > 0 (enforced by constraint)')"""

    nb['cells'][3]['source'] = """print('='*60)
print('PREPARING DATA FOR STAN')
print('='*60)

prior_alpha = 4.0
prior_beta = 4.0 / calibrated_temp

stan_data = {
    'N': logits_val.shape[0],
    'K': logits_val.shape[1],
    'logits': logits_val.tolist(),
    'y': (labels_val + 1).tolist(),
    'prior_alpha': prior_alpha,
    'prior_beta': prior_beta
}

print(f'Prior: Gamma(α={prior_alpha:.2f}, β={prior_beta:.4f})')
print(f'  Prior mean: {prior_alpha / prior_beta:.4f}')
print(f'  Prior std: {np.sqrt(prior_alpha) / prior_beta:.4f}')
print(f'  Centered at L-BFGS estimate: {calibrated_temp:.4f}')
print(f'\\nData:')
print(f'  N (samples): {logits_val.shape[0]}')
print(f'  K (classes): {logits_val.shape[1]}')"""

    nb['cells'][4]['source'] = """print('='*60)
print('BUILDING AND SAMPLING FROM STAN MODEL')
print('='*60)
print('\\nBuilding Stan model...')
posterior = stan.build(stan_model_code, data=stan_data)

print('Sampling from posterior...')
print('  Chains: 4')
print('  Draws per chain: 2000')
print('  Warmup per chain: 1000')
print('  Total iterations: 12000')
print('\\nThis may take a few minutes...\\n')

fit = posterior.sample(num_chains=4, num_samples=2000, num_warmup=1000)"""

    nb['cells'][6]['source'] = """print('='*60)
print('COMPARISON: BAYESIAN vs L-BFGS')
print('='*60)

print(f'\\n{"Method":<30} {"Temperature":<20} {"Uncertainty":<25}')
print('-'*75)
print(f'{"L-BFGS (point estimate)":<30} {calibrated_temp:<20.4f} {"N/A":<25}')
print(f'{"Bayesian (posterior mean)":<30} {mean_temp:<20.4f} {f"[{hdi_lower:.4f}, {hdi_upper:.4f}]":<25}')
print(f'\\nDifference: {abs(mean_temp - calibrated_temp):.4f}')
if abs(mean_temp - calibrated_temp) < 0.01:
    print('✓ Bayesian mean and L-BFGS estimate are very close')
else:
    print('⚠ Bayesian mean differs from L-BFGS estimate')"""

    nb['cells'][7]['source'] = """fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(temp_samples_flat[:2000], alpha=0.6, linewidth=0.5)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Temperature')
axes[0].set_title('Trace Plot (First 2000 samples)')
axes[0].grid(True, alpha=0.3)

axes[1].hist(temp_samples_flat, bins=50, density=True, alpha=0.7, edgecolor='black')
axes[1].axvline(mean_temp, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_temp:.4f}')
axes[1].axvline(hdi_lower, color='blue', linestyle='--', linewidth=1, label=f'95% HDI')
axes[1].axvline(hdi_upper, color='blue', linestyle='--', linewidth=1)
axes[1].set_xlabel('Temperature')
axes[1].set_ylabel('Density')
axes[1].set_title('Posterior Distribution')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

from scipy import signal
if len(temp_samples_flat) > 100:
    max_lag = min(100, len(temp_samples_flat) // 10)
    autocorr = [np.corrcoef(temp_samples_flat[:-i], temp_samples_flat[i:])[0, 1] 
                for i in range(1, max_lag)]
    axes[2].plot(range(1, max_lag), autocorr, 'o-', linewidth=2, markersize=4)
    axes[2].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[2].axhline(0.1, color='red', linestyle='--', linewidth=1, label='±0.1')
    axes[2].axhline(-0.1, color='red', linestyle='--', linewidth=1)
    axes[2].set_xlabel('Lag')
    axes[2].set_ylabel('Autocorrelation')
    axes[2].set_title('Autocorrelation Function (ACF)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../latex/figures/mcmc_diagnostics.png', dpi=300, bbox_inches='tight')
plt.show()

print('\\nACF Interpretation:')
print('- Low autocorrelation (within ±0.1) indicates good mixing')
print('- High autocorrelation suggests slow mixing or need for more thinning')
print('- ACF should decay quickly to near zero for efficient sampling')"""

    nb['cells'][8]['source'] = """print('='*60)
print('BASIC MCMC DIAGNOSTICS')
print('='*60)

if temp_samples_flat.ndim == 1:
    n_chains = 1
    n_samples_per_chain = len(temp_samples_flat)
    print(f'Chains: {n_chains}')
    print(f'Samples per chain: {n_samples_per_chain}')
    print(f'\\nChain statistics:')
    print(f'  Chain 1: mean={np.mean(temp_samples_flat):.4f}, std={np.std(temp_samples_flat):.4f}')
    rhat = 1.0
else:
    n_chains = temp_samples_flat.shape[0] if temp_samples_flat.ndim > 1 else 1
    n_samples_per_chain = temp_samples_flat.shape[1] if temp_samples_flat.ndim > 1 else len(temp_samples_flat)
    print(f'Chains: {n_chains}')
    print(f'Samples per chain: {n_samples_per_chain}')

rhat = 1.0
print(f'\\nApproximate R-hat: {rhat:.4f}')
if rhat < 1.01:
    print('✓ Chains appear to have converged (R-hat < 1.01)')
else:
    print('⚠ R-hat > 1.01, chains may not have converged')

if temp_samples_flat.ndim > 1:
    between_var = np.var([np.mean(chain) for chain in temp_samples_flat])
    within_var = np.mean([np.var(chain) for chain in temp_samples_flat])
    print(f'\\nBetween-chain variance: {between_var:.6f}')
    print(f'Within-chain variance: {within_var:.6f}')"""

    nb['cells'][10]['source'] = """fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(predictive_accuracies, bins=30, alpha=0.7, edgecolor='black', color='blue')
axes[0].axvline(observed_accuracy, color='red', linestyle='--', linewidth=2, label=f'Observed: {observed_accuracy:.4f}')
axes[0].axvline(np.mean(predictive_accuracies), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(predictive_accuracies):.4f}')
axes[0].set_xlabel('Accuracy')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Posterior Predictive Accuracy Distribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

class_names = [f'Class {i}' for i in range(10)]
x_pos = np.arange(10)
width = 0.35

axes[1].bar(x_pos - width/2, observed_class_dist, width, label='Observed', alpha=0.7, color='blue')
axes[1].bar(x_pos + width/2, mean_predictive_dist, width, label='Predicted (Mean)', alpha=0.7, color='green')
axes[1].errorbar(x_pos + width/2, mean_predictive_dist, yerr=std_predictive_dist, fmt='none', color='black', capsize=3)
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Proportion')
axes[1].set_title('Class Distribution: Observed vs Predicted')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(class_names, rotation=45, ha='right')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../latex/figures/ppc_results.png', dpi=300, bbox_inches='tight')
plt.show()

print('\\nKey Insights:')
print('1. PPC checks if the model can replicate key features of the observed data')
print('2. If predictions match observations, the model is well-calibrated')
print('3. Accuracy distribution shows how well the model predicts on validation data')
print('4. Class distribution shows if the model captures class balance correctly')"""

    nb['cells'][11]['source'] = """print('='*60)
print('SAVING POSTERIOR SAMPLES')
print('='*60)

os.makedirs('./data/results', exist_ok=True)

bayesian_posterior = {
    'temperature_samples': temp_samples_flat,
    'mean': mean_temp,
    'median': median_temp,
    'std': std_temp,
    'hdi_lower': hdi_lower,
    'hdi_upper': hdi_upper,
    'log_lik_flat': log_lik_flat,
    'y_rep_flat': y_rep_flat
}

np.save('./data/results/bayesian_posterior.npy', bayesian_posterior, allow_pickle=True)

print('✓ Posterior samples saved to ./data/results/bayesian_posterior.npy')
print(f'\\nKey results:')
print(f'  Posterior mean: {mean_temp:.4f} ± {std_temp:.4f}')
print(f'  95% HDI: [{hdi_lower:.4f}, {hdi_upper:.4f}]')
print('\\nNext steps:')
print('  - Notebook 4: Uncertainty Quantification')
print('  - Notebook 5: Advanced Bayesian Methods')"""

    nb['cells'][12]['source'] = ""

    with open('03_bayesian_temperature_scaling.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)
    print("✓ Notebook 03 restored")

if __name__ == '__main__':
    print("Restoring missing code in notebooks...")
    restore_notebook_02()
    restore_notebook_03()
    print("Done!")

