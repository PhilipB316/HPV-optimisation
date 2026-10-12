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

import os
import matplotlib.pyplot as plt
import numpy as np

from .models import MODELS

FITTED = ("cp", "exp", "biexp")  # the disproportionate model shares the cp fit

class Rider:
    """A rider's mean-maximal-power (MMP) curve and the models fitted to it.

    After `fit`, `models` holds the fitted physiological models:
        "cp": CPModel, critical power and W'.
        "exp": ExpModel, single-exponential power ceiling.
        "biexp": BiExpModel, bi-exponential power ceiling.
        "disproportionate": DisproportionateModel, shares the "cp" fit.
    Fitted values are in `model.params`.
    """
    def __init__(self, times, watts):
        """
        Args:
            times: MMP durations in s.
            watts: best average power in W for each duration.
        """
        order = np.argsort(times)
        self.times = np.asarray(times, float)[order]
        self.watts = np.asarray(watts, float)[order]
        self.models = {name: cls() for name, cls in MODELS.items()}

    def fit(self):
        """Fit every model to the MMP data.

        Returns:
            The Rider, so calls can be chained: `Rider(t, w).fit()`.
        """

        for model in self.models.values():
            model.fit(self.times, self.watts)
        return self

    def fit_quality(self):
        """Print and return how well each model fits the MMP data.

        Returns:
            dict of model name ("cp", "exp", "biexp") to a dict with `rss`,
            `rmse`, `mape`, `max_pct` and per-point `pct_err`.
        """
        t, actual = self.times, self.watts
        n = len(actual)

        print(f"{'Model':<12}{'RSS':>12}{'RMSE':>10}{'MAPE%':>9}{'maxErr%':>10}")
        results = {}
        for name in FITTED:
            resid = actual - self.models[name].predict(t)
            pct_err = np.abs(resid) / actual * 100

            rss = float(np.sum(resid ** 2))
            rmse = float(np.sqrt(rss / n))
            mape = float(np.mean(pct_err))
            max_pct = float(np.max(pct_err))

            results[name] = dict(rss=rss, rmse=rmse, mape=mape, max_pct=max_pct, pct_err=pct_err)
            print(f"{name:<12}{rss:>12.0f}{rmse:>10.1f}{mape:>9.2f}{max_pct:>10.2f}")
        return results

    def plot_fits(self, show=True, savefig=None):
        """Plot the MMP data with the fitted model curves.

        Args:
            show: call plt.show().
            savefig: file path to save a PDF to, or None.

        Returns:
            The matplotlib Axes.
        """
        plot_t = np.linspace(min(self.times), max(self.times), 500)
        colors = {"cp": "red", "exp": "blue", "biexp": "green"}

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(self.times, self.watts, color="black", zorder=5, label="MMP data")
        for name in FITTED:
            model = self.models[name]
            t = plot_t[plot_t >= model.valid_from]
            ax.plot(t, model.predict(t), color=colors[name],
                    label=f"{name} (CP={model.cp:.0f}W)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Power (W)")
        ax.legend()
        fig.tight_layout()
        if savefig:
            plt.savefig(os.path.splitext(savefig)[0] + '.pdf')
        if show:
            plt.show()
        return ax

