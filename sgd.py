import numpy as np
import matplotlib.pyplot as plt

# Parameters
n, d = 1000, 1000
num_iterations = 200
learning_rate = 0.01
var_epsilons = [1, 0.1, 0.01, 0]
# Adjust parameters for A in R^(10000 x 1000)
n_large, d = 10000, 1000

# Function to generate A and epsilon
def generate_A_and_epsilon(n, d, var_epsilon):
    A = np.random.normal(0, 1/np.sqrt(d), (n, d))
    epsilon = np.random.normal(0, np.sqrt(var_epsilon), n)
    return A, epsilon

# Function to generate large A with specific row variance
def generate_large_A_and_epsilon(n, d, var_epsilon):
    A = np.zeros((n, d))
    for j in range(n):
        A[j, :] = np.random.normal(0, 1/np.sqrt(1000 * (j+1)), d)
    epsilon = np.random.normal(0, np.sqrt(var_epsilon), n)
    return A, epsilon

# SGD implementation
def stochastic_gradient_descent(A, b, num_iterations, learning_rate):
    w = np.zeros(A.shape[1])
    losses = []
    # Perform SGD for num_iterations
    for i in range(num_iterations):
        idx = np.random.randint(0, A.shape[0])
        Ai = A[idx, :]
        bi = b[idx]
        gradient = Ai.T.dot(Ai.dot(w) - bi)
        w = w - learning_rate * gradient
        loss = 0.5 * np.linalg.norm(A.dot(w) - b)**2
        losses.append(loss)
    
    return w, losses

# Helper function to run experiments for different variances of epsilon
def run_experiments(n, d, num_iterations, learning_rate, var_epsilons, A_large):
    for i, var_epsilon in enumerate(var_epsilons):
        A, epsilon = generate_large_A_and_epsilon(n, d, var_epsilon) if A_large else generate_A_and_epsilon(n, d, var_epsilon)
        b = A.dot(np.ones(d)) + epsilon
        w, losses = stochastic_gradient_descent(A, b, num_iterations, learning_rate)

        axs[i].plot(losses)
        axs[i].set_title(f'SGD Convergence with Variance of Epsilon = {var_epsilon}')
        axs[i].set_xlabel('Iteration')
        axs[i].set_ylabel('Loss')

    plt.tight_layout()
    plt.show()

# Initialize plots for the original A
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

# Run experiments for different variances of epsilon with the original A
run_experiments(n, d, num_iterations, learning_rate, var_epsilons, False)

# Initialize plots for the adjusted large A
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

# Run experiments for different variances of epsilon with the adjusted larger A
run_experiments(n_large, d, num_iterations, learning_rate, var_epsilons, True)
