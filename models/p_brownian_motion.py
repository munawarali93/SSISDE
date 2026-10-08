model = "Psi_approx_BS"
#model = "Psi_true_BS"
BP = np.zeros(K)
dt = T/K
t = np.linspace(0, T, K)
for i in range(1,K):
  BP[i] = BP[i-1] + BQ[i] - BQ[i-1] -  (mu_pred1[i-1]/sigma_pred[i-1])*dt
