sigma_prime_ = diff_method._differentiate(sigma_pred_.reshape(-1, 1), S_).flatten()
sigma_2prime_ = diff_method._differentiate(sigma_prime_.reshape(-1, 1), S_).flatten()
sigma_3prime_ = diff_method._differentiate(sigma_2prime_.reshape(-1, 1), S_).flatten()