import numpy as np

m = 10
n = 12


master_matrix = np.random.random((m, m))
master_matrix_inv = np.linalg.inv(master_matrix)

sigma_cup = [2, 5, 6, 8]
tau_cup = [0, 1, 6]



N_inv = np.delete(np.delete(master_matrix_inv, sigma_cup, axis = 0), sigma_cup, axis = 1) - np.take(np.delete(master_matrix_inv, sigma_cup, axis = 0), sigma_cup, axis = 1) @ np.linalg.inv(np.take(np.take(master_matrix_inv, sigma_cup, axis = 0), sigma_cup, axis = 1)) @ np.delete(np.take(master_matrix_inv, sigma_cup, axis = 0), sigma_cup, axis = 1)

print("Is outer reduction successfu?", np.all(np.round(N_inv, 4) == np.round(np.linalg.inv(np.delete(np.delete(master_matrix, sigma_cup, axis = 0), sigma_cup, axis = 1)), 4)))
