import numpy as np

def fitting_error(normal, proxy_N, proxy_theta):
    dot = np.dot(normal, proxy_N)
    raw_error = (dot - np.cos(proxy_theta)) ** 2
    

    return max(raw_error, 0.001)  # Reduced from 0.05 to 0.001

def boundary_promotion(chart, triangle, mesh):
    shared_edges = 0  
    return shared_edges / (3 - shared_edges)  

def compute_cost(F, C, P, alpha=1, beta=0.7, gamma=0.5):

    # Handle potential numerical issues
    F = max(F, 1e-6)
    C = max(C, 1e-6)
    P = max(P, 1e-6)
    
    return (F**alpha) * (C**beta) * (P**gamma)
