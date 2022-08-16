import numpy as np
from scipy.stats import norm


data = np.array([
    [55, 141],
    [33, 145],
    [70, 139],
    [49, 161]
])

# Test 1, Wald per week on fractions.
print('Test 1, Wald test per week with BH corrected p-values')
total = data.sum(axis=0)
fc = data[:, 0] / total[0]
fj = data[:, 1] / total[1]
f_hat = fc - fj
se_hat = np.sqrt(fc * (1 - fc) / total[0] + fj * (1 - fj) / total[1])

wald = f_hat / se_hat
p_values = 2 * norm.cdf(-abs(wald))

# Benjamini-Hochberg correction
alpha = 0.05
step = 1 / data.shape[0]
r = np.argwhere(np.sort(p_values) < alpha * np.arange(step, 1.0 + step, step))[-1][0]
threshold = np.sort(p_values)[r]

for w, p in zip([-2, -1, 1, 2], p_values):
    print(f'Week {w:>2} p-value {p:.4f} reject {p <= threshold}')
