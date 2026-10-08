QV = np.r_[0.0, np.cumsum(np.diff(S_)**2)]
diff_method = ps.FiniteDifference(order=3)
dQV = diff_method._differentiate(QV.reshape(-1, 1), t).flatten()
