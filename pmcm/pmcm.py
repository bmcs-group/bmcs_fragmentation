"""
Probabilistic multiple cracking model
"""
from typing import List, Any, Union
import bmcs_utils.api as bu

import traits.api as tr
import numpy as np
import warnings

from bmcs_utils.trait_types import Float
from scipy.optimize import newton

warnings.filterwarnings("error", category=RuntimeWarning)


class CrackBridgeModel(bu.Model):
    """
    Record of all material parameters of the composite. The model components
    (PullOutModel, CrackBridgeRespSurf, PMCM) are all linked to the database record
    and access the parameters they require. Some parameters are shared across all
    three components (Em, Ef, vf), some are specific to a particular type of the
    PulloutModel.
    """

class CBMConstantBond(CrackBridgeModel):
    """
    Return the matrix stress profile of a crack bridge for a given control slip
    at the loaded end
    """
    Em = bu.Float(28000)
    Ef = bu.Float(180000)
    vf = bu.Float(0.01)
    T = bu.Float(8)

    ipw_view = bu.View(
        bu.Item('Em'),
        bu.Item('Ef'),
        bu.Item('vf'),
        bu.Item('T'),
    )

    @property
    def Ec(self):
        return self.Em * (1 - self.vf) + self.Ef * self.vf  # [MPa] mixture rule

class CrackBridgeRespSurface(bu.Model):
    """
    Crack bridge response surface that returns the values of matrix stress
    along ahead of a crack and crack opening for a specified remote stress
    and boundary conditions.
    """
    cb = bu.Instance(CrackBridgeModel)

    def get_sig_m(self, sig_c: float, z: np.ndarray) -> np.ndarray:
        """Get the profile of matrix stress along the specimen
        :param z: np.ndarray
        :type sig_c: float
        """
        cb = self.cb
        sig_m = np.minimum(z * cb.T * cb.vf / (1 - cb.vf), cb.Em * sig_c /
                           (cb.vf * cb.Ef + (1 - cb.vf) * cb.Em))
        return sig_m

    def get_eps_f(self, sig_c: float, z: np.ndarray) -> np.ndarray:
        cb = self.cb
        sig_m = self.get_sig_m(sig_c, z )
        eps_f = (sig_c - sig_m * (1 - cb.vf)) / cb.vf / cb.Ef
        return eps_f

    sig_c_slider: float = bu.Float(1.0, BC=True)

    ipw_view = bu.View(
        bu.Item('sig_c_slider')
    )

    @staticmethod
    def subplots(fig):
        ax = fig.subplots(1,1)
        ax1 = ax.twinx()
        return ax, ax1

    def update_plot(self, axes):
        ax, ax1 = axes
        x_range = np.linspace(-10,10,10)
        z_range = np.abs(x_range)

        sig_m_range = self.get_sig_m(self.sig_c_slider, z_range)
        eps_f_range = self.get_eps_f(self.sig_c_slider, z_range)

        ax.plot(x_range, sig_m_range, color='black')
        ax.fill_between(x_range, sig_m_range, color='gray', alpha=0.1)
        ax1.plot(x_range, eps_f_range, color='blue')
#        ax1.fill_between(x_range, eps_f_range, color='blue', alpha=0.1)



class PMCM(bu.Model):
    name = "PMCM"
    """
    Implement the global crack tracing algorithm based on a crack bridge response surface
    """
    cb = bu.Instance(CrackBridgeModel)
    def _cb_default(self):
        return CBMConstantBond()

    cb_rs = tr.Property(depends_on="state_changed")
    @tr.cached_property
    def _get_cb_rs(self):
        return CrackBridgeRespSurface(cb=self.cb)

    tree = ['cb', 'cb_rs']

    n_x = bu.Int(5000)
    L_x = bu.Float(500)
    sig_cu = bu.Float(20)
    sig_mu = bu.Float(10)
    m = bu.Float(4)

    ipw_view = bu.View(
        bu.Item('n_x'),
        bu.Item('L_x'),
        bu.Item('sig_cu'),
        bu.Item('sig_mu'),
        bu.Item('m'),
    )

    def get_z_x(self, x, XK):  # distance to the closest crack (*\label{get_z_x}*)
        """Specimen discretization
        """
        z_grid = np.abs(x[:, np.newaxis] - np.array(XK)[np.newaxis, :])
        return np.amin(z_grid, axis=1)

    def get_sig_c_z(self, sig_mu, z, sig_c_pre):
        """
        :param sig_c_pre:
        :type sig_mu: float
        """
        # crack initiating load at a material element
        #print('sig_mu', sig_mu)
        fun = lambda sig_c: sig_mu - self.cb_rs.get_sig_m(sig_c, z)
        try:  # search for the local crack load level
            sig_c = newton(fun, sig_c_pre)
            #print('sig_c', sig_c)
            return sig_c
        except (RuntimeWarning, RuntimeError):
            # solution not found (shielded zone) return the ultimate composite strength
            return self.sig_cu

    def get_sig_c_K(self, z_x, x, sig_c_pre, sig_mu_x):
        # crack initiating loads over the whole specimen
        get_sig_c_x = np.vectorize(self.get_sig_c_z)
        sig_c_x = get_sig_c_x(sig_mu_x, z_x, sig_c_pre)
        #print('sig_c_x', z_x, x, sig_c_pre, sig_mu_x)
        #print('sig_c_x', sig_c_x)
        y_idx = np.argmin(sig_c_x)
        return sig_c_x[y_idx], x[y_idx]

    def get_cracking_history(self, update_progress=None):
        cb = self.cb
        L_x, n_x, sig_mu, sig_cu, m = self.L_x, self.n_x, self.sig_mu, self.sig_cu, self.m
        x = np.linspace(0, L_x, n_x)  # specimen discretization
        sig_mu_x: np.ndarray[np.float_] = sig_mu * np.random.weibull(
            m, size=n_x)  # matrix strength

        XK: List[float] = []  # recording the crack postions
        sig_c_K: List[float] = [0.]  # recording the crack initiating loads
        eps_c_K: List[float] = [0.]  # recording the composite strains
        CS: List[float] = [L_x, L_x / 2]  # initial crack spacing
        sig_m_x_K: List[float] = [np.zeros_like(x)]  # stress profiles for crack states

        Ec: float = cb.Ec
        Em: float = cb.Em
        idx_0 = np.argmin(sig_mu_x)
        XK.append(x[idx_0])  # position of the first crack
        sig_c_0 = sig_mu_x[idx_0] * Ec / Em
        #print('sig_c_0', sig_c_0)
        sig_c_K.append(sig_c_0)
        eps_c_K.append(sig_mu_x[idx_0] / Em)

        while True:
            z_x = self.get_z_x(x, XK)  # distances to the nearest crack
            sig_m_x_K.append(self.cb_rs.get_sig_m(sig_c_K[-1], z_x))  # matrix stress
            sig_c_k, y_i = self.get_sig_c_K(z_x, x, sig_c_K[-1], sig_mu_x)  # identify next crack
            if sig_c_k == sig_cu:
                break
            if update_progress:  # callback to user interface
                update_progress(sig_c_k)
            XK.append(y_i)  # record crack position
            sig_c_K.append(sig_c_k)  # corresponding composite stress
            eps_c_K.append(  # composite strain - integrate the strain field
                np.trapz(self.cb_rs.get_eps_f(sig_c_k, self.get_z_x(x, XK)), x) / np.amax(x))
            XK_arr = np.hstack([[0], np.sort(np.array(XK)), [L_x]])
            CS.append(np.average(XK_arr[1:] - XK_arr[:-1]))  # crack spacing

        sig_c_K.append(sig_cu)  # the ultimate state
        eps_c_K.append(np.trapz(self.cb_rs.get_eps_f(sig_cu, self.get_z_x(x, XK) ), x) / np.amax(x))
        CS.append(CS[-1])
        if update_progress:
            update_progress(sig_c_k)
        return np.array(sig_c_K), np.array(eps_c_K), sig_mu_x, x, np.array(CS), np.array(sig_m_x_K)

    @staticmethod
    def subplots(fig):
        ax1, ax2 = fig.subplots(1,2)
        ax11 = ax1.twinx()
        return ax1, ax11, ax2

    def plot(self, axes):
        ax, ax_cs, ax_sig_x = axes
        sig_c_K, eps_c_K, sig_mu_x, x, CS, sig_m_x_K = self.get_cracking_history()
        n_c = len(eps_c_K) - 2  # numer of cracks
        ax.plot(eps_c_K, sig_c_K, marker='o', label='%d cracks:' % n_c)
        ax.set_xlabel(r'$\varepsilon_\mathrm{c}$ [-]');
        ax.set_ylabel(r'$\sigma_\mathrm{c}$ [MPa]')
        ax_sig_x.plot(x, sig_mu_x, color='orange')
        ax_sig_x.fill_between(x, sig_mu_x, 0, color='orange', alpha=0.1)
        ax_sig_x.set_xlabel(r'$x$ [mm]');
        ax_sig_x.set_ylabel(r'$\sigma$ [MPa]')
        ax.legend()
        eps_c_KK = np.array([eps_c_K[:-1], eps_c_K[1:]]).T.flatten()
        CS_KK = np.array([CS[:-1], CS[:-1]]).T.flatten()
        ax_cs.plot(eps_c_KK, CS_KK, color='gray')
        ax_cs.fill_between(eps_c_KK, CS_KK, color='gray', alpha=0.2)
        ax_cs.set_ylabel(r'$\ell_\mathrm{cs}$ [mm]');

    def update_plot(self, axes):
        self.plot(axes)
