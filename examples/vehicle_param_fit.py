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

vehicle = hv.Vehicle(rider_mass=80, bike_mass=55, cda=0.030, crr=0.008,
                     air_density=1.2, slope=-0/150, wind=0)

run1 = hv.Run("./data/sample_data_2.tcx",
              start=200, end=423, smooth_window=3)

run1.to_csv("./data/run_1.csv")

run2 = hv.Run("./data/sample_data_2.tcx",
              start=920, end=1192, smooth_window=3)
run3 = hv.Run("./data/sample_data_1.tcx",
              end=250, smooth_window=3)
run4 = hv.Run("./data/sample_data_1.tcx",
              start=520,end=765, smooth_window=3)

kakahi = vehicle.fit([run1, run2, run3, run4])

print(kakahi)
print("Power at 100 km/hr:", kakahi.total_power_consumed(100))

kakahi.simulate(run1).plot(savefig="../docs/fit.png")
kakahi.simulate(run2).plot()
kakahi.simulate(run3).plot()
kakahi.simulate(run4).plot()

