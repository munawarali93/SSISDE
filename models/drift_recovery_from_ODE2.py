A = 2*sigma_prime/sigma_pred
B = sigma_2prime/sigma_pred + 2*(sigma_prime)**2/(sigma_pred)**2


mu_pred11 = np.zeros(K)
lamda_pred = np.zeros(K)
mu_pred11[0] = mu_0
lamda_pred[0] = lambda_0 if "lambda_0" in globals() else r
for i in range(1,len(a)):
  mu_pred11[i] = mu_pred11[i-1] + lamda_pred[i-1]*(S[i] - S[i-1])
  lamda_pred[i] = lamda_pred[i-1] + (A[i-1]*lamda_pred[i-1] + B[i-1]*mu_pred11[i-1]) * (S[i] - S[i-1])
