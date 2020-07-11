"""
Probabilistic multiple cracking model
"""

import traits.api as tr
import numpy as np
import warnings

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
        """

        :type sig_c: np.ndarray
        """
        mp = self.mp
        sig_m = np.minimum(z * mp.T * mp.vf / (1 - mp.vf), mp.Em * sig_c /
                           (mp.vf * mp.Ef + (1 - mp.vf) * mp.Em))
        return sig_m

    def get_eps_f(self, sig_c):  # reinforcement strain (*\label{sig_f}*)
        sig_m = get_sig_m(z, sig_c)
        eps_f = (sig_c - sig_m * (1 - vf)) / vf / Ef
        return eps_f

    @property
    def _get_Ec(self):
        return self.Em * (1 - self.vf) + self.Ef * self.vf  # [MPa] mixture rule


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
        # crack initiating load at a material element
        fun = lambda sig_c: sig_mu - self.cb_rs.get_sig_m(z, sig_c)
        try:  # search for the local crack load level
            return newton(fun, sig_c_pre)
        except (RuntimeWarning, RuntimeError):
            # solution not found (shielded zone) return the ultimate composite strength
            return sig_cu

    def get_sig_c_K(self, z_x, x, sig_c_pre, sig_mu_x):
        # crack initiating loads over the whole specimen
        get_sig_c_x = np.vectorize(self.get_sig_c_z)
        sig_c_x = get_sig_c_x(sig_mu_x, z_x, sig_c_pre)
        y_idx = np.argmin(sig_c_x)
        return sig_c_x[y_idx], x[y_idx]

    def get_cracking_history(self, update_progress=None):
        x = np.linspace(0, self.L_x, self.n_x)  # specimen discretization (*\label{discrete}*)
        sig_mu_x = self.sig_mu * np.random.weibull(m, size=self.n_x)  # matrix strength (*\label{m_strength}*)

        XK = []  # recording the crack postions
        sig_c_K = [0.]  # recording the crack initating loads
        eps_c_K = [0.]  # recording the composite strains
        CS = [self.L_x, self.L_x / 2]  # initial crack spacing
        sig_m_x_K = [np.zeros_like(x)]  # stress profiles for crack states

        Ec: float = self.cb_rs.Ec
        Em: float = self.cb_rs.Em
        idx_0 = np.argmin(sig_mu_x)
        XK.append(x[idx_0])  # position of the first crack
        sig_c_0 = sig_mu_x[idx_0] * Ec / Em
        sig_c_K.append(sig_c_0)
        eps_c_K.append(sig_mu_x[idx_0] / Em)

        while True:
            z_x = self.get_z_x(x, XK)  # distances to the nearest crack
            sig_m_x_K.append(self.cb_rs.get_sig_m(z_x, sig_c_K[-1]))  # matrix stress
            sig_c_k, y_i = self.get_sig_c_K(z_x, x, sig_c_K[-1], sig_mu_x)  # identify next crack
            if sig_c_k == self.sig_cu:  # (*\label{no_crack}*)
                break
            if update_progress:  # callback to user interface
                update_progress(sig_c_k)
            XK.append(y_i)  # record crack position
            sig_c_K.append(sig_c_k)  # corresponding composite stress
            eps_c_K.append(  # composite strain - integrate the strain field
                np.trapz(self.cb_rs.get_eps_f(self.get_z_x(x, XK), sig_c_k), x) / np.amax(x))
            XK_arr = np.hstack([[0], np.sort(np.array(XK)), [self.L_x]])
            CS.append(np.average(XK_arr[1:] - XK_arr[:-1]))  # crack spacing

        sig_c_K.append(self.sig_cu)  # the ultimate state
        eps_c_K.append(np.trapz(self.cb_rs.get_eps_f(self.get_z_x(x, XK), self.sig_cu), x) / np.amax(x))
        CS.append(CS[-1])
        if update_progress:
            update_progress(sig_c_k)
        return np.array(sig_c_K), np.array(eps_c_K), sig_mu_x, x, np.array(CS), np.array(sig_m_x_K)
