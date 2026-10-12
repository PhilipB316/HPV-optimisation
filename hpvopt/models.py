# Copyright (C) 2026 Philip Brand
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

class Model:
    p_max = 2000       # realised power is clipped to this
    valid_from = 0     # shortest duration (s) the fitted curve is valid for

    def __init__(self, **params):
        self.params = params

    @property
    def cp(self):
        return self.params["CP"]


class CPModel(Model):
    """Model 1: critical power and W' (P = CP + W'/t).

    params: CP (W), W_prime (J). Fitted to the 60-1800 s part of the MMP
    data. Power is capped at 1000 W. W' recovers below CP using the
    Bartram et al. time constant.
    """
    p_max = 1000
    valid_from = 60

    def fit(self, t, p):
        """Linearised hyperbolic fit: P = W'/t + CP, using 60-1800 s only."""
        mask = (t >= 60) & (t <= 1800)
        W_prime, CP, rvalue, _, _ = stats.linregress(1.0 / t[mask], p[mask])
        self.params = {"CP": CP, "W_prime": W_prime, "r": rvalue}
        return self

    def predict(self, t):
        return self.params["CP"] + self.params["W_prime"] / t

    def realise(self, power_profile, dt):
        """
        Depletion (P > CP):  dW'/dt = -(P - CP)
        Recovery  (P < CP):  dW'/dt = (W' - W'_bal) / tau_W'
        """
        CP, W_prime = self.params["CP"], self.params["W_prime"]
        w_bal = W_prime
        realised = np.empty(len(power_profile))
        for i, p_raw in enumerate(power_profile):
            p = min(max(p_raw, 0), self.p_max)

            if p > CP:
                wbal_consumed = min((p - CP) * dt, w_bal)
                p = CP + wbal_consumed / dt
                w_bal -= wbal_consumed
            elif p < CP:
                tau_w = 2287.2 * (CP - p) ** -0.688
                w_bal += (W_prime - w_bal) * (1.0 - np.exp(-dt / tau_w))

            realised[i] = p
        return realised


class ExpModel(Model):
    """Model 2: single-exponential power ceiling (P = (Pmax - CP)exp(-t/tau) + CP).

    params: Pmax (W), tau (s), CP (W). Recovery uses tau_rec = 3 tau, which is
    a placeholder.
    """
    def fit(self, t, p):
        def model(t, Pmax, tau, CP):
            return (Pmax - CP) * np.exp(-t / tau) + CP

        CP0 = p[-1] * 0.95
        A0 = p[0] - CP0
        tau0 = t[len(t) // 2]
        popt, _ = curve_fit(
            model, t, p, p0=(A0 + CP0, tau0, CP0),
            bounds=([0, 1, 0], [np.inf, np.inf, np.inf]),
            maxfev=10000,
        )
        self.params = dict(zip(["Pmax", "tau", "CP"], popt))
        return self

    def predict(self, t):
        Pmax, tau, CP = (self.params[k] for k in ("Pmax", "tau", "CP"))
        return (Pmax - CP) * np.exp(-t / tau) + CP

    def realise(self, power_profile, dt):
        """
        State: s in [0, 1] (freshness; 1 = fully fresh).
        Ceiling:    P_ceil(s) = (Pmax - CP) * s + CP
        Depletion (P > CP):  ds/dt = -(P - CP) / ((Pmax - CP) * tau)
        Recovery  (P <= CP): ds/dt = (1 - s) / tau_rec,  tau_rec = 3 * tau (placeholder)
        """
        Pmax, tau, CP = (self.params[k] for k in ("Pmax", "tau", "CP"))
        tau_rec = tau * 3.0
        s = 1.0
        realised = np.empty(len(power_profile))
        amplitude = Pmax - CP
        for i, p_raw in enumerate(power_profile):
            p = float(np.clip(p_raw, 0, self.p_max))
            p_ceil = amplitude * s + CP
            p = min(p, p_ceil)
            realised[i] = p

            if p > CP:
                ds = -(p - CP) / (amplitude * tau) * dt
            else:
                ds = (1.0 - s) / tau_rec * dt
            s = float(np.clip(s + ds, 0.0, 1.0))
        return realised


class BiExpModel(Model):
    """Model 3: bi-exponential power ceiling.

    P = A1 exp(-t/tau1) + A2 exp(-t/tau2) + CP.
    params: A1, tau1, A2, tau2, CP. Fast and slow components deplete above CP
    and do not recover.
    """

    def fit(self, t, p):
        def model(t, A1, tau1, A2, tau2, CP):
            return A1 * np.exp(-t / tau1) + \
                   A2 * np.exp(-t / tau2) + CP

        CP0 = p[-1] * 0.95
        A1_0 = p[0] - p[len(p) // 3]
        A2_0 = p[len(p) // 3] - CP0
        popt, _ = curve_fit(
            model, t, p, p0=(A1_0, 15, A2_0, 300, CP0),
            bounds=([0, 1, 0, 50, 0], [np.inf, 60, np.inf, 2000, np.inf]),
            maxfev=20000,
        )
        self.params = dict(zip(["A1", "tau1", "A2", "tau2", "CP"], popt))
        return self

    def predict(self, t):
        A1, tau1, A2, tau2, CP = (self.params[k] for k in ("A1", "tau1", "A2", "tau2", "CP"))
        return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + CP

    def realise(self, power_profile, dt):
        """
        Ceiling: P_ceil(s1, s2) = CP + A1*s1 + A2*s2
        Above CP each tank depletes in proportion to its share of the ceiling.
        There is no recovery.
        """
        A1, tau1, A2, tau2, CP = (self.params[k] for k in ("A1", "tau1", "A2", "tau2", "CP"))
        s1, s2 = 1.0, 1.0
        realised = np.empty(len(power_profile))

        for i, p_raw in enumerate(power_profile):
            p = float(np.clip(p_raw, 0, self.p_max))
            p_ceil = CP + A1 * s1 + A2 * s2
            p = min(p, p_ceil)
            realised[i] = p

            if p > CP:
                excess = p - CP
                denom = A1 * s1 + A2 * s2
                frac1 = (A1 * s1) / denom if denom > 1e-9 else 0.5
                ds1 = -(frac1 * excess) / (A1 * tau1) * dt
                ds2 = -((1 - frac1) * excess) / (A2 * tau2) * dt
            else:
                ds1 = 0
                ds2 = 0

            s1 = float(np.clip(s1 + ds1, 0.0, 1.0))
            s2 = float(np.clip(s2 + ds2, 0.0, 1.0))
        return realised


class DisproportionateModel(CPModel):
    """Model 4: disproportionate W' expenditure at high power (Fitton et al., Model III).

    Uses the CP and W' fitted for Model 1. Pmax is `p_max` (1000 W). W' does
    not recover.
    """
    p_max = 1000

    def realise(self, power_profile, dt):
        CP, W_prime = self.params["CP"], self.params["W_prime"]
        w_bal = W_prime
        realised = np.empty(len(power_profile))

        for i, p_raw in enumerate(power_profile):
            p = min(max(p_raw, 0), self.p_max)

            scale = (self.p_max - CP) / (self.p_max - p) if p < self.p_max else np.inf

            if p > CP:
                wbal_consumed = min((p - CP) * scale * dt, w_bal)
                p = CP + wbal_consumed / (scale * dt)
                w_bal -= wbal_consumed
            realised[i] = p
        return realised

MODELS = {
    "cp": CPModel,
    "exp": ExpModel,
    "biexp": BiExpModel,
    "disproportionate": DisproportionateModel,
}

