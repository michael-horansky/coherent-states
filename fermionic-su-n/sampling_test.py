
import numpy as np
import functions



N = 4 # We sampling N component random vector
N_sample = 10000

# Actual sampling parameters
means = np.array([1.0, 10.0, -5.0, 2.0])
std = np.array([2.0, 3.0, 6.0, 1.0])
pearson_cor = np.identity(N)
pearson_cor[0][2] = 0.2
pearson_cor[2][0] = 0.2
pearson_cor[0][3] = 0.8
pearson_cor[3][0] = 0.8
pearson_cor[1][2] = -0.4
pearson_cor[2][1] = -0.4

# Covariance and product means
Sigma = pearson_cor * np.outer(std, std)
product_means = Sigma + np.outer(means, means) #[i][j] = E[X_i X_j]



# We add some covariance -- must be symmetrical!
"""
product_means[0][2] += 2.0
product_means[2][0] += 2.0

product_means[0][3] += 0.5
product_means[3][0] += 0.5

product_means[1][2] -= 20
product_means[2][1] -= 20"""

print(f"Expected values:")
for i in range(N):
    print(f"  E[X_{i + 1}] = {means[i]}")

print(f"Expected stds:")
for i in range(N):
    print(f"  sigma[X_{i + 1}] = {std[i]}")

print(f"Expected values of products:")
for i in range(N):
    for j in range(i, N):
        print(f"  E[X_{i + 1} X_{j + 1}] = {product_means[i][j]}")

# Make sigma
eigvals, eigvecs = np.linalg.eigh(Sigma)
min_eigval = min(eigvals)
print(f"Minimum covariance matrix eigenvalue = {min_eigval}")



# Now we take our samples
sample = functions.sample_with_autocorrelation(means, Sigma, N_sample)

measured_means = np.zeros(N)
measured_stds = np.zeros(N)
measured_product_means = np.zeros((N, N))

measured_means = np.average(sample, axis = 0)
measured_stds = np.std(sample, axis = 0)

for n in range(N_sample):
    for i in range(N):
        for j in range(i, N):
            measured_product_means[i][j] += sample[n][i] * sample[n][j]
measured_product_means /= N_sample



print(f"Measured values:")
for i in range(N):
    print(f"  E[X_{i + 1}] = {measured_means[i]:0.4f} (comp. w/ {means[i]})")

print(f"Measured stds:")
for i in range(N):
    print(f"  sigma[X_{i + 1}] = {measured_stds[i]:0.4f} (comp. w/ {std[i]})")

print(f"Measured values of products:")
for i in range(N):
    for j in range(i, N):
        print(f"  E[X_{i + 1} X_{j + 1}] = {measured_product_means[i][j]:0.4f} (comp. w/ {product_means[i][j]})")


