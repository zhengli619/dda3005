import numpy as np
import matplotlib.pyplot as plt
import time

# Assuming clustering function from clustering.py is available
def clustering(X, Z0=None, opts={'iter': 100, 'display': True}):
    n, m = X.shape
    if Z0 is None:
        k = 3
        Z0 = np.random.randn(n, k)
    else:
        k = Z0.shape[1]
    Z = Z0

    opts_alg = {}
    opts_alg['iter'] = opts.get('iter', 100)
    opts_alg['display'] = opts.get('display', True)

    sX = np.tile(X, (k, 1))

    for i in range(opts_alg['iter']):
        sZ = np.abs(sX - np.tile(Z.flatten('F').reshape(n * k, 1, order='F'), (1, m))) ** 2
        sZ = np.sqrt(sum(sZ.reshape(n, m * k, order='F')))
        sZ = sZ.reshape(k, m, order='F')
        mZ = np.min(sZ, axis=0)
        c = np.argmin(sZ, axis=0)
        for j in range(k):
            ind = (c == j)
            Z[:, j] = np.mean(X[:, ind], axis=1)

    if opts_alg['display']:
        s1 = np.sum(np.sum(mZ ** 2))
        s2 = s1 / (m * k)
        print('Total sum of distances = {:5.2f}; Weighted sum of distances = {:1.6e}'.format(s1, s2))

    return c, Z

# Function to generate random data clouds A, B, C
def generate_data(p):
    A = np.random.normal(0, 1, (2, p))
    B = 4 + np.random.normal(0, 1, (2, p))
    C = np.array([[5], [2]]) + np.random.normal(0, 1, (2, p))
    return np.hstack((A, B, C))  # Combine A, B, C to form X

# Set up data sizes for different runs
p_values = [100 * i for i in range(1, 11)]  # p = 100, 200, ..., 1000
mean_times = []

# Run clustering for different values of p and measure time
for p in p_values:
    X = generate_data(p)
    Z0 = np.random.randn(2, 3)  # Initial points for 3 clusters

    start_time = time.time()
    _, Z_final = clustering(X, Z0, opts={'iter': 100, 'display': False})
    end_time = time.time()

    elapsed_time = end_time - start_time
    mean_times.append(elapsed_time)

# Plot mean CPU time per iteration
plt.figure(figsize=(10, 6))
plt.plot(p_values, mean_times, marker='o')
plt.xlabel('Data Size (p)')
plt.ylabel('Mean CPU Time (s)')
plt.title('Mean CPU Time per Iteration for Different Data Sizes')
plt.grid(True)
plt.show()

# Visualize clustering result for the largest p
X = generate_data(p_values[-1])  # Use the largest p
Z0 = np.random.randn(2, 3)  # Initial points for 3 clusters

# Run clustering
c, Z_final = clustering(X, Z0, opts={'iter': 100, 'display': False})

# Visualize the clusters
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b']
for j in range(3):
    plt.scatter(X[0, c == j], X[1, c == j], c=colors[j], label=f'Cluster {j+1}', alpha=0.5)

# Plot the final cluster centers
plt.scatter(Z_final[0, :], Z_final[1, :], c='k', marker='x', s=100, label='Cluster Centers')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Clustering Results for Largest p')
plt.legend()
plt.grid(True)
plt.show()