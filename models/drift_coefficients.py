mu_0 = mu_exact[0]
#a  = sigma_prime/sigma_pred.squeeze()
#b  = (2*l21 + sigma_pred.squeeze()*sigma_prime**2 + sigma_pred.squeeze()**2*sigma_2prime)/(2*sigma_pred.squeeze())
#G = mu_prime*sigma_pred - mu_exact*sigma_prime - 0.5*(sigma_pred*sigma_prime**2 + sigma_pred**2*sigma_2prime)

a  = sigma_prime/sigma_pred
b  = l21/sigma_pred + 0.5*(sigma_prime**2 + sigma_pred*sigma_2prime)
G = - 0.5*(sigma_pred*sigma_prime**2 + sigma_pred**2*sigma_2prime)
b_  = G/sigma_pred + 0.5*(sigma_prime**2 + sigma_pred*sigma_2prime)