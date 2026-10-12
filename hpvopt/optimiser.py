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

import cma
import numpy as np
from .run import Run
from .vehicle import fastest_200m_segment

class OptimisationResult:
    """The optimised power profile and the speed it achieves.

    Attributes:
        run: Run with the optimised `watts` and simulated `calc_speeds`.
        speed: fastest 200 m average speed in km/h.
    """
    def __init__(self, run, speed):
        self.run = run # Run with the optimised watts and calc_speeds
        self.speed = speed  # fastest 200 m average speed, km/h

    def plot(self, **kwargs):
        """Plot the optimised run. Arguments are passed to `hpvopt.plot`."""
        return self.run.plot(**kwargs)

    def save(self, path):
        """Save the optimised run to a .npz file."""
        self.run.save(path)

    @classmethod
    def load(cls, path, vehicle):
        """Load a profile saved with `save`.

        The speed is recomputed for `vehicle`, so loading with a different
        vehicle gives that vehicle's speed for the same power profile.

        Args:
            path: .npz file.
            vehicle: Vehicle used to compute `speed`.
        """
        run = Run(path, label="Optimised")
        dt = run.times[1] - run.times[0]
        t, d = Optimiser(vehicle, None, dt=dt).distance_profile(run.watts)
        return cls(run, fastest_200m_segment(t, d)[0] * 3.6)

class Optimiser:
    """Finds the power profile that maximises the fastest 200 m average speed.

    The profile is searched with CMA-ES, with realised power limited by the
    physiological model.

    Args:
        vehicle: Vehicle to simulate.
        model: a fitted model from `Rider.models`.
        duration: length of the profile in s.
        dt: time step in s.
        bounds: (min, max) power in W for the search.
        roughness_weighting: penalty on power changes between steps.
        time_weighting: penalty on power drawn late in the run.
        road_length_limit: stop the simulation after this many m, or None.
    """
    def __init__(self, vehicle, model, duration=600, dt=2, bounds=(0, 2000),
                 roughness_weighting=3e-7, time_weighting=3e-5,
                 road_length_limit=None):
        self.vehicle = vehicle
        self.model = model
        self.duration = duration
        self.dt = dt
        self.bounds = bounds
        self.roughness_weighting = roughness_weighting
        self.time_weighting = time_weighting
        self.road_length_limit = road_length_limit  # m, None to disable
        self.n = int(duration / dt)
        self.time_weights = np.linspace(0.0, 1.0, self.n)

    def penalties(self, power_profile, max_idx):
        roughness = np.sum(np.diff(power_profile[:max_idx]) ** 2)
        roughness_p = roughness * self.roughness_weighting

        time_p = np.sum(self.time_weights[:max_idx] *
                        power_profile[:max_idx]) * self.time_weighting

        return roughness_p + time_p

    def distance_profile(self, realised):
        vehicle = self.vehicle
        current_speed = 0.0
        current_energy = 0.0
        current_distance = 0.0
        p_prev = realised[0]
        times = [0.0]
        distances = [0.0]
        index = 0
        for p in realised:
            index += 1
            step_energy_produced = 0.5 * (p_prev + p) * \
                self.dt * vehicle.drive_train_efficiency
            step_energy_consumed = vehicle.total_power_consumed(current_speed) * self.dt
            current_energy += (step_energy_produced - step_energy_consumed)
            if current_energy < 0:
                current_energy = 0
            prev_speed = current_speed
            current_speed = vehicle.current_speed_from_energy(current_energy)

            # advance cumulative distance (trapezoidal, km/h -> m/s)
            current_distance += 0.5 * (prev_speed + current_speed) / 3.6 * self.dt
            times.append(index * self.dt)
            distances.append(current_distance)
            p_prev = p

            if self.road_length_limit is not None and current_distance >= self.road_length_limit:
                break
        return times, distances

    def evaluate(self, power_profile):
        """
        Fitness function: negative of the fastest average speed (km/h) over any
        200m segment, plus roughness/time penalties on the portion of the power
        profile used to reach the end of that segment.
        """
        realised = self.model.realise(power_profile, self.dt)
        times, distances = self.distance_profile(realised)

        best_avg_speed_m_s, _, best_end_time = fastest_200m_segment(times, distances)
        best_speed = best_avg_speed_m_s * 3.6

        if best_end_time is None:
            best_index = len(times) - 1
        else:
            best_index = min(int(round(best_end_time / self.dt)), len(times) - 1)

        return -best_speed + self.penalties(realised, best_index)

    def run(self, n_jobs=0, verbose=1):
        """Run the optimisation.

        Args:
            n_jobs: number of worker processes. 0 runs serially.
            verbose: CMA-ES output level. 0 is silent.

        Returns:
            An OptimisationResult. Power after the fastest 200 m segment is
            not constrained by the penalties and should be ignored.
        """
        x0 = [self.model.cp] * self.n
        sigma0 = 10

        es = cma.CMAEvolutionStrategy(x0, sigma0, {
            'bounds': list(self.bounds),
            'maxiter': 1000,
            'popsize': 25,
            'tolx': 1e-6,
            'tolfun': 0.03,
            'verbose': verbose
        })
        es.optimize(self.evaluate, n_jobs=n_jobs)

        best_power = self.model.realise(es.result.xbest, self.dt)
        times = np.arange(0, self.duration, self.dt)

        run = Run.from_arrays(times, best_power, label="Optimised")
        run.calc_speeds = np.array(self.vehicle.calculate_speeds(times, best_power))

        t, d = self.distance_profile(best_power)
        speed = fastest_200m_segment(t, d)[0] * 3.6
        return OptimisationResult(run, speed)

