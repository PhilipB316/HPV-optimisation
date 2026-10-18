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
                     air_density=1.2, slope=-0/150, wind=0)

run1 = hv.Run("../../data/20261019_Clint_KakahiShell.tcx",
              end=250, smooth_window=5)
run2 = hv.Run("../../data/20261019_Clint_KakahiShell.tcx",
              start=520,end=765, smooth_window=3)
run3 = hv.Run("../../data/20261019_Clint_KakahiShell_02.tcx",
              start=200, end=423, smooth_window=5)
run4 = hv.Run("../../data/20261019_Clint_KakahiShell_02.tcx",
              start=920, end=1192, smooth_window=3)

# sim = vehicle.simulate(run1)
# sim.plot()
# sim = vehicle.simulate(run2)
# sim.plot()

fitted_vehicle = vehicle.fit([run1, run2, run3, run4])
# fitted_vehicle.slope=-20/5000
# fitted_vehicle.bike_mass = 25
# fitted_vehicle.wind = 1.6
# fitted_vehicle.cda = 0.03

print(fitted_vehicle)
print("power at 100 km/hr")
print(fitted_vehicle.total_power_consumed(100))

sim = fitted_vehicle.simulate(run2)
sim.plot()

