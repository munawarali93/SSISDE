sigma_prime = diff_method._differentiate(sigma_pred.reshape(-1, 1), S).flatten()
sigma_2prime = diff_method._differentiate(sigma_prime.reshape(-1, 1), S).flatten()
sigma_3prime = diff_method._differentiate(sigma_2prime.reshape(-1, 1), S).flatten()