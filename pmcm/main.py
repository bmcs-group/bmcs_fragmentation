import numpy as np
import matplotlib.pylab as plt
import pmcm

mp = pmcm.ModelParams(m=100,T=1, sig_mu=3)
cb = pmcm.CrackBridgeRespSurface(mp=mp)
mc = pmcm.PMCM(mp=mp, cb_rs=cb)

if False:
    z = np.linspace(-2, 2, 100)
    sig_m = cb.get_sig_m(z, 10)
    plt.plot(z, sig_m)
    plt.show()

fig, (ax, ax_sig_x) = plt.subplots(1,2,figsize=(8,3),tight_layout=True)
ax_cs = ax.twinx()
mc.plot(ax, ax_cs, ax_sig_x)
plt.show()