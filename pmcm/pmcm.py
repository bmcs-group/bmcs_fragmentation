"""
Probabilistic multiple cracking model
"""
from typing import List, Any, Union

import traits.api as tr
import numpy as np
import warnings
from scipy.optimize import newton

warnings.filterwarnings("error", category=RuntimeWarning)


class ModelParams(tr.HasTraits):
    """
    Record of all material parameters of the composite. The model components
    (PullOutModel, CrackBridgeRespSurf, PMCM) are all linked to the database record
    and access the parameters they require. Some parameters are shared across all
    three components (Em, Ef, vf), some are specific to a particular type of the
    PulloutModel.
    """
    Em = tr.Float(28000)
    Ef = tr.Float(180000)
    vf = tr.Float(0.01)
    T = tr.Float(8)
    n_x = tr.Int(5000)
    L_x = tr.Float(500)
    sig_cu = tr.Float(20)
    sig_mu = tr.Float(10)
    m = tr.Float(4)

    @property
    def Ec(self):
        return self.Em * (1 - self.vf) + self.Ef * self.vf  # [MPa] mixture rule


class PullOutModel(tr.HasTraits):
    """
    Return the matrix stress profile of a crack bridge for a given control slip
    at the loaded end
    """
    mp = tr.Instance(ModelParams)


class CrackBridgeRespSurface(tr.HasTraits):
    """
    Crack bridge response surface that returns the values of matrix stress
    along ahead of a crack and crack opening for a specified remote stress
    and boundary conditions.
    """
    mp = tr.Instance(ModelParams)

    def get_sig_m(self, sig_c: float, z: np.ndarray) -> np.ndarray:
        """Get the profile of matrix stress along the specimen
        :param z: np.ndarray
        :type sig_c: float
        """
        mp = self.mp
        sig_m = np.minimum(z * mp.T * mp.vf / (1 - mp.vf), mp.Em * sig_c /
                           (mp.vf * mp.Ef + (1 - mp.vf) * mp.Em))
        return sig_m

    def get_eps_f(self, sig_c: float, z: np.ndarray) -> np.ndarray:
        mp = self.mp
        sig_m = self.get_sig_m(z, sig_c)
        eps_f = (sig_c - sig_m * (1 - mp.vf)) / mp.vf / mp.Ef
        return eps_f


class PMCM(tr.HasTraits):
    """
    Implement the global crack tracing algorithm based on a crack bridge response surface
    """
    mp = tr.Instance(ModelParams)

    cb_rs = tr.Instance(CrackBridgeRespSurface)

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
        mp = self.mp
        # crack initiating load at a material element
        print('sig_mu', sig_mu)
        fun = lambda sig_c: sig_mu - self.cb_rs.get_sig_m(z, sig_c)
        try:  # search for the local crack load level
            sig_c = newton(fun, sig_c_pre)
            print('sig_c', sig_c)
            return sig_c
        except (RuntimeWarning, RuntimeError):
            # solution not found (shielded zone) return the ultimate composite strength
            return mp.sig_cu

    def get_sig_c_K(self, z_x, x, sig_c_pre, sig_mu_x):
        # crack initiating loads over the whole specimen
        get_sig_c_x = np.vectorize(self.get_sig_c_z)
        sig_c_x = get_sig_c_x(sig_mu_x, z_x, sig_c_pre)
        print('sig_c_x', z_x, x, sig_c_pre, sig_mu_x)
        print('sig_c_x', sig_c_x)
        y_idx = np.argmin(sig_c_x)
        return sig_c_x[y_idx], x[y_idx]

    def get_cracking_history(self, update_progress=None):
        mp = self.mp
        x = np.linspace(0, mp.L_x, mp.n_x)  # specimen discretization
        sig_mu_x: np.ndarray[np.float_] = mp.sig_mu * np.random.weibull(
            mp.m, size=mp.n_x)  # matrix strength

        XK: List[float] = []  # recording the crack postions
        sig_c_K: List[float] = [0.]  # recording the crack initiating loads
        eps_c_K: List[float] = [0.]  # recording the composite strains
        CS: List[float] = [mp.L_x, mp.L_x / 2]  # initial crack spacing
        sig_m_x_K: List[float] = [np.zeros_like(x)]  # stress profiles for crack states

        Ec: float = mp.Ec
        Em: float = mp.Em
        idx_0 = np.argmin(sig_mu_x)
        XK.append(x[idx_0])  # position of the first crack
        sig_c_0 = sig_mu_x[idx_0] * Ec / Em
        print('sig_c_0', sig_c_0)
        sig_c_K.append(sig_c_0)
        eps_c_K.append(sig_mu_x[idx_0] / Em)

        while True:
            print('xxxxxxxxxxxxxxxxxxxxxxx')
            z_x = self.get_z_x(x, XK)  # distances to the nearest crack
            print('sig_c_K', sig_c_K)
            sig_m_x_K.append(self.cb_rs.get_sig_m(z_x, sig_c_K[-1]))  # matrix stress
            sig_c_k, y_i = self.get_sig_c_K(z_x, x, sig_c_K[-1], sig_mu_x)  # identify next crack
            print(sig_c_k, y_i)
            if sig_c_k == mp.sig_cu:
                break
            if update_progress:  # callback to user interface
                update_progress(sig_c_k)
            XK.append(y_i)  # record crack position
            sig_c_K.append(sig_c_k)  # corresponding composite stress
            eps_c_K.append(  # composite strain - integrate the strain field
                np.trapz(self.cb_rs.get_eps_f(self.get_z_x(x, XK), sig_c_k), x) / np.amax(x))
            XK_arr = np.hstack([[0], np.sort(np.array(XK)), [mp.L_x]])
            CS.append(np.average(XK_arr[1:] - XK_arr[:-1]))  # crack spacing

        sig_c_K.append(mp.sig_cu)  # the ultimate state
        eps_c_K.append(np.trapz(self.cb_rs.get_eps_f(self.get_z_x(x, XK), mp.sig_cu), x) / np.amax(x))
        CS.append(CS[-1])
        if update_progress:
            update_progress(sig_c_k)
        return np.array(sig_c_K), np.array(eps_c_K), sig_mu_x, x, np.array(CS), np.array(sig_m_x_K)

    def plot(self, ax, ax_cs, ax_sig_x):
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
