import stan
import httpstan
import nest_asyncio
nest_asyncio.apply()
print(httpstan.__version__)

# Simple test model
model_code = """
data {
    int<lower=0> N;
    array[N] real y;
}
parameters {
    real mu;
    real<lower=0> sigma;
}
model {
    y ~ normal(mu, sigma);
}
"""

# Test data
data = {'N': 5, 'y': [1.0, 2.0, 3.0, 4.0, 5.0]}

# Build model
posterior = stan.build(model_code, data=data)

# Sample from posterior
fit = posterior.sample(num_chains=1, num_samples=100)

print("Success! PyStan is working.")