import numpy as np
import matplotlib.pylab as plt
import pmcm

mp = pmcm.ModelParams()
po = pmcm.CrackBridgeRespSurface(mp=mp)

z = np.linspace(-2, 2, 100)
sig_m = po.get_sig_m(z, 10)
plt.plot(z, sig_m)
plt.show()
