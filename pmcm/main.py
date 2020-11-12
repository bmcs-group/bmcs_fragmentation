import numpy as np
import matplotlib.pylab as plt
import pmcm

mp = pmcm.ModelParams(
    Em=25e3,  # [MPa] matrix modulus
    Ef=180e3,  # [MPa] fiber modulus
    vf=0.01,  # [-] reinforcement ratio
    T=12.,  # [N/mm^3] bond intensity
    sig_cu=10.0, # [MPa] composite strength
    sig_mu=3.0, # [MPa] matrix strength
    m=10000 # Weibull shape modulus
)

cb = pmcm.CrackBridgeRespSurface(mp=mp)
mc = pmcm.PMCM(mp=mp, cb_rs=cb)

if False:
    z = np.linspace(-2, 2, 100)
    sig_m = cb.get_sig_m(z, 10)
    plt.plot(z, sig_m)
    plt.show()

print('yyyyyyyyyyy')
fig, (ax, ax_sig_x) = plt.subplots(1, 2, figsize=(8, 3), tight_layout=True)
ax_cs = ax.twinx()
mc.plot(ax, ax_cs, ax_sig_x)
plt.show()
