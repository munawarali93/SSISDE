drift =  "BS"
diff = "BS"
#Change to "BS" for Black-Scholes Model
def mu(x,drift):
  if drift == "constant":
    return r*np.ones_like(x)
  if drift == "BS":
    return r*x
  if drift == "quadratic":
    return r*x**2
  if drift == "quadratic1":
    return  0.3*x**2 + 0.7*x
  if drift == "quadratic2":
    return  0.25*x*(1.0-x*x)
  if drift == "sin":
    return r*np.sin(x)
  if drift == "OU":
    return kappa * (theta - x)
  if drift == "Double_well_potential":
    return w1*x**3 + w2*x
  if drift == "population":
    return r*x*(K + x)

def sigma(x,diff):
  if diff == "OU":
    return sigm*np.ones_like(x)
  if diff == "BS":
    return sigm*x
  if diff == "quadratic":
    return sigm*x**2
  if diff == "quadratic1":
    return   0.1*x**2 + 0.5*x
  if diff == "quadratic2":
    return   -0.5*(1.0-x**2)
  if diff == "Double_well_potential":
    return w3 + w4*x**2
  if diff == "population":
    return beta*x

S_ = np.zeros(N)
S_[0]=S0
for i in range(N-1):
  S_[i+1] = S_[i] + mu(S_[i],drift) * dt + (sigma(S_[i],diff)) * dW[i]

mu_exact_ = mu(S_,drift)
sigma_exact_ = sigma(S_,diff)

