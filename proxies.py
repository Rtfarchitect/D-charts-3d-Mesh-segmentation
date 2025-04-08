import numpy as np
from scipy.optimize import minimize

def compute_proxy(normals, weights):
    """Compute proxy (N, theta) for a set of normals and weights."""
    def objective(params):
        N, theta = params[:3], params[3]
        N_normalized = N / np.linalg.norm(N)
        errors = (np.dot(normals, N_normalized) - np.cos(theta))**2
        return np.sum(weights * errors)
    
    # Initial guess: mean normal and 0 angle
    initial_guess = np.append(np.mean(normals, axis=0), 0)
    result = minimize(objective, initial_guess, constraints={'type': 'eq', 'fun': lambda x: np.linalg.norm(x[:3]) - 1})
    N = result.x[:3] / np.linalg.norm(result.x[:3])
    theta = result.x[3]
    return N, theta