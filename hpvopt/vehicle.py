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

import math, copy
import numpy as np
from dataclasses import dataclass, replace
from scipy.optimize import minimize

BOUNDS = {'crr': (0.001, 0.2), 'cda': (0.01, 0.9), 'air_density': (0.8, 2.0)}

def fastest_200m_segment(times, distances):
    """
    Find the fastest 200m segment of the ride, using linear interpolation
    between samples
    Returns:
        (avg_speed_m_s, start_time, end_time) for the fastest 200m segment,
    """
    SEGMENT = 200.0
    n = len(distances)
    best_avg_speed_m_s = 0.0
    best_start_time = None
    best_end_time = None
    j = 0
    for i in range(n):
        target = distances[i] + SEGMENT
        if j < i:
            j = i
        while j < n and distances[j] < target:
            j += 1
        if j >= n:
            break  # not enough remaining distance to complete a 200m segment

        # interpolate the time at which cumulative distance crosses `target`
        if j == i:
            t_end = times[j]
        else:
            d0, d1 = distances[j - 1], distances[j]
            t0, t1 = times[j - 1], times[j]
            frac = 0.0 if d1 <= d0 else (target - d0) / (d1 - d0)
            frac = min(max(frac, 0.0), 1.0)
            t_end = t0 + frac * (t1 - t0)

        segment_time = t_end - times[i]
        if segment_time > 0:
            avg_speed_m_s = SEGMENT / segment_time
            if avg_speed_m_s > best_avg_speed_m_s:
                best_avg_speed_m_s = avg_speed_m_s
                best_start_time = times[i]
                best_end_time = t_end

    return best_avg_speed_m_s, best_start_time, best_end_time

@dataclass
class Vehicle:
    """Power-balance model of the vehicle and rider.

    Args:
        rider_mass: kg.
        bike_mass: kg.
        cda: drag coefficient times frontal area, m^2.
        crr: rolling resistance coefficient.
        air_density: kg/m^3.
        slope: road gradient in m/m, negative is downhill.
        wind: m/s, positive is a tailwind.
        drive_train_efficiency: fraction of rider power reaching the wheels.
        wheel_mass: kg per wheel, used for rotational kinetic energy.
        number_of_wheels: number of wheels.
        gravity: m/s^2.
    """
    rider_mass: float               # kg
    bike_mass: float                # kg
    cda: float                      # m^2
    crr: float
    air_density: float = 1.1        # kg/m^3
    slope: float = 0.0              # m/m, negative is downhill
    wind: float = 0.0               # m/s, positive is tail wind
    drive_train_efficiency: float = 0.97
    wheel_mass: float = 2           # kg per wheel
    number_of_wheels: int = 3
    gravity: float = 9.81           # m/s^2

    def __str__(self):
        return (f"Vehicle\n"
                f"  mass          {self.rider_mass} + {self.bike_mass} kg (rider + bike)\n"
                f"  CdA           {self.cda:.4f} m^2\n"
                f"  Crr           {self.crr:.5f}\n"
                f"  air density   {self.air_density:.3f} kg/m^3\n"
                f"  slope         {self.slope:.4f}\n"
                f"  wind          {self.wind:.2f} m/s (+ = tailwind)\n"
                f"  efficiency    {self.drive_train_efficiency}")

    def drag_power(self, speed):
        """Calculate power lost to aerodynamic drag at a given speed in km/h."""
        relative_speed = speed / 3.6 - self.wind
        return 0.5 * self.air_density * self.cda * relative_speed ** 3

    def rolling_resistance_power(self, speed):
        """Calculate power lost to rolling resistance at a given speed in km/h."""
        total_mass = self.rider_mass + self.bike_mass
        return self.crr * total_mass * self.gravity * (speed / 3.6)

    def slope_power(self, speed):
        """Calculates power asisted by gravity on a slope at a given speed in km/h."""
        total_mass = self.rider_mass + self.bike_mass
        return total_mass * self.gravity * math.sin(math.atan(self.slope)) * (speed / 3.6)

    def total_power_consumed(self, speed):
        """Calculate total power consumed at a given speed in km/hr."""
        return self.drag_power(speed) + self.rolling_resistance_power(speed) + self.slope_power(speed)

    def current_speed_from_energy(self, kinetic_energy):
        """Calculates current speed in km/h from kinetic linear and rotational energy."""
        total_mass = self.rider_mass + self.bike_mass
        kinetic_energy_demon = (0.5 * total_mass + 0.25 * self.number_of_wheels * self.wheel_mass)
        return np.sqrt(kinetic_energy / kinetic_energy_demon) * 3.6

    def calculate_speeds(self, times, watts, initial_speed=0.0):
        """
        Calculates the speed of the vehicle in km/hr through all time in 'times' array
        Returns len(times) - 1 speeds, one per step between samples.
        """
        effective_mass = 0.5*(self.rider_mass+self.bike_mass) + \
            0.25*self.number_of_wheels*self.wheel_mass
        current_speed = initial_speed
        current_energy = effective_mass * (initial_speed/3.6)**2

        time_step = times[1] - times[0]
        current_speed = 0
        current_energy = 0
        calc_speeds = []
        for i in range(len(times) - 1):
            step_energy_produced = 0.5 * (watts[i] + watts[i+1]) * time_step * self.drive_train_efficiency
            step_energy_consumed = self.total_power_consumed(current_speed) * time_step
            current_energy += (step_energy_produced - step_energy_consumed)
            if current_energy < 0:
                current_energy = 0
            if i > 0 and (times[i] - times[i-1]) > time_step:
                current_energy = 0
            current_speed = self.current_speed_from_energy(current_energy)
            calc_speeds.append(current_speed)
        return calc_speeds

    def simulate(self, run):
        """Simulate the speed produced by the power in `run`.

        Args:
            run: Run with `times` and `watts`.

        Returns:
            A copy of `run` with `calc_speeds` set, in km/h. It has one fewer
            sample than `times`, since each value is the speed after a step.
            The original run is unchanged.
        """
        sim = copy.copy(run)
        sim.calc_speeds = np.array(self.calculate_speeds(run.times, run.watts))
        return sim

    def fit(self, runs, free=('cda', 'crr'), bounds=None):
        """Fit vehicle parameters to measured speed.

        Args:
            runs: list of Run objects with measured `speeds`.
            free: names of the parameters to fit.
            bounds: dict of name -> (low, high), overriding the defaults.

        Returns:
            A new Vehicle with the fitted parameters. The original is unchanged.

        Example:
            fitted = vehicle.fit([run1, run2], free=("cda", "crr"))
        """
        bounds = {**BOUNDS, **(bounds or {})}
        targets = []
        for run in runs:
            speeds = np.asarray(run.speeds, float)
            targets.append(speeds)

        def objective(params):
            total_squared_error = 0
            total_count = 0
            for run, speeds in zip(runs, targets):
                vehicle = replace(self, **dict(zip(free, params)))
                if run.rider_mass is not None:
                    vehicle = replace(vehicle, rider_mass=run.rider_mass)
                sim = np.array(vehicle.calculate_speeds(run.times, run.watts,
                                                        initial_speed=speeds[0]))
                rec = speeds[1:]
                n = min(len(sim), len(rec))
                total_squared_error += np.sum((sim[:n] - rec[:n]) ** 2)
                total_count += n
            return np.sqrt(total_squared_error / total_count)

        initial_guess = [getattr(self, k) for k in free]
        result = minimize(objective, initial_guess,
                          bounds=[bounds.get(k, (None, None)) for k in free],
                          method='L-BFGS-B')
        return replace(self, **dict(zip(free, result.x)))

