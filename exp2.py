import numpy as np
import matplotlib.pyplot as plt
import mlp_module
from mlp_module import truncate_string



"""generation of samples from stable distribution with a given alpha and beta and fixed mu = 0 and sigma = 1"""
alpha = 0.7; beta = 0; N = 1000
X = mlp_module.stable_sample(alpha, beta, N)





"""generation of samples from stable distribution with a given beta and fixed alpha, mu = 0 and sigma = 1. 
   The following two lines of code produce a sample and estimate its beta"""
alpha = 1.5; beta = 0.4; N = 5000
T = [-4, -1, 0, 1, 5]  # this is the T vector used for estimating beta.
mlp_module.beta_estimate_error_functions(T, alpha, beta, N, 1)





"""Estimation of alpha"""
X = mlp_module.stable_sample(alpha, beta, N)
# Check of symmetry
y_min_max = mlp_module.symmetry_deviation(len(X), 1.1, 100)
F0 = mlp_module.empirical_distribution_function(X, 0)
symmetric = mlp_module.symmetry_identifier(F0, y_min_max)
# Choice of grid function for sas
grid_function = mlp_module.grid_search_choice(8)
T = np.arange(0, 3, 0.05)
T = grid_function(T)
if max(T) < 0.999:
    T[-1] = 0.999
# parameter estimation
if symmetric:
    alpha_est = mlp_module.mlp_ebs_sas(X, T)
    print(f'alpha = {alpha}, alpha_est = {truncate_string(alpha_est)}')
alpha = 1.7
test_beta = 1
X = mlp_module.stable_sample(alpha, test_beta, N)
Y = X.copy()
Z = []
for j in range(4):
    np.random.shuffle(Y)     #usage of bootstrap
    half_Y = int(len(Y)/2)
    Z = Z + list(Y[:half_Y] -  Y[half_Y:])
Z = np.array(Z)
Z = Z/2**(1/alpha)
alpha_est1 = mlp_module.mlp_ebs_sas(X, T)
alpha_est2 = mlp_module.mlp_ebs_sas(Z, T)
print(f'(trua_alpha, alpha_X, alpha_Z) = ({alpha}, {truncate_string(alpha_est1)},{truncate_string(alpha_est2)})')


