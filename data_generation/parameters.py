N = 100000
T = 1.0
t = np.linspace(0, T, N)
dt = T / N

np.random.seed(3)
dW = np.sqrt(dt) * np.random.randn(N)
W_ = np.cumsum(dW)


r = 0.5
sigm = 0.3
S0 = 1.0
kappa = 0.5
theta = 0
w1 = -1
w2 = 0.5
w3 = 1.0
w4 = 0.25
K = 0.1
beta = 0.2