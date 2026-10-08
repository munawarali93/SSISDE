mu_pred1 = np.zeros(K)
mu_pred2 = np.zeros(K)
mu_pred1[0] = mu_0
mu_pred2[0] = mu_0
for i in range(1,len(a)):
  mu_pred1[i] = mu_pred1[i-1] + (a[i-1]*mu_pred1[i-1]+ b[i-1])*(S[i] - S[i-1])
  mu_pred2[i] = mu_pred2[i-1] + (a[i-1]*mu_pred2[i-1]+ b_[i-1])*(S[i] - S[i-1])
mu_pred2 = np.round(mu_pred2,8)