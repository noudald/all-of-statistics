import numpy as np

from scipy.stats import norm


# Exercise 10.7.a

x0 = [.225, .262, .217, .240, .230, .229, .235, .217]
x1 = [.209, .205, .196, .210, .202, .207, .224, .223, .220, .201]

delta_hat = np.mean(x0) - np.mean(x1)
var_hat = np.var(x0) / len(x0) + np.var(x1) / len(x1)
se_hat = np.sqrt(var_hat)

W = delta_hat / se_hat

print(f'P-value: {2 * norm.cdf(-abs(W)):.5f}')
print(f'Confidence interval: ({delta_hat - 2*se_hat:.3f}, {delta_hat + 2*se_hat:.3f})')


# Exercise 10.7.b

num_simulations = 10**5
rng = np.random.RandomState(37)

data = np.concatenate([x0, x1])
t_obs = abs(delta_hat)
t_perms = np.empty([num_simulations])
for i in range(num_simulations):
    rng.shuffle(data)
    x = data[0:len(x0)]
    y = data[len(x0):]
    t_perms[i] = abs(np.mean(x) - np.mean(y))

print(f'P-value: {np.sum(t_perms > t_obs) / num_simulations:.5f}')


