M = 100
K = int(N/M)
l21 = []
for i in tqdm(range(K)):
  path = np.array([[t[k], BQ_[k]] for k in range(i*M,i*M+M)])
  X1 = np.array([iisignature.sig(path[0:i], 3) for i in range(1,len(path)+1)])
  X1_reduced = X1[:, keep_cols]
  clf = linear_model.Ridge(alpha=0.01)
  clf.fit(X1_reduced, S_[i*M:i*M+M])
  l21.append(clf.coef_[3])
l21 = np.array(l21)
