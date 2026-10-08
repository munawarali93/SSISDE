BQ_ = np.zeros(len(S_))
for i in range(1,len(S_)):
  BQ_[i] = BQ_[i-1] + (S_[i] - S_[i-1])/sigma_pred_[i-1]
