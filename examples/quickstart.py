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

import hpvopt as hv

mmp_times = [1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 1800, 3600]
mmp_watts = [1288, 1278, 1268, 1218, 792, 558, 477, 296, 244, 242, 228, 220, 204]

vehicle = hv.Vehicle(rider_mass=80, bike_mass=55, cda=0.030, crr=0.008,
                     air_density=1.1, slope=-1 / 150, wind=4 / 3.6)

run = hv.Run("../../data/20261008_Clint_KakahiShell.tcx", start=2890, end=3100)
sim = vehicle.simulate(run)
sim.plot()

fitted = vehicle.fit([run])
print(fitted)

rider = hv.Rider(mmp_times, mmp_watts).fit()
rider.fit_quality()
rider.plot_fits()

result = hv.Optimiser(fitted, rider.models["biexp"], duration=600, dt=2).run(n_jobs=4)
print(result.speed)
result.plot()
result.save("profile.npz")

result = hv.OptimisationResult.load("profile.npz", vehicle)
result.plot()

attempt = hv.Run("../../../data/20261008_Clint_KakahiShell.tcx", start=2890, end=3100)

hv.plot(result.run,
        series=[["watts"], ["watts", "calc_speeds"]],
        labels=[{"watts": "Optimised Power"},
                {"watts": "Measured Power", "calc_speeds": "Simulated Speed"}])
