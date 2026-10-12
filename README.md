# HPV-optimisation

A general tool-kit for human powered vehicle development.

Contains features for:

- Vehicle parameter fitting to run data
- Vehicle speed prediction from previous run data
- Optimisation of future power profiles based on vehicle specifications and
physiological models and rider MMP data

## Install

Requires Python 3.10 or later.

```bash
git clone https://github.com/PhilipB316/HPV-optimisation
cd HPV-optimisation
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Dependencies are numpy, scipy, matplotlib and cma.

## Overview

| Object | Purpose |
|---|---|
| `Run` | Time series of power and speed, loaded from a file or built from arrays |
| `Vehicle` | Power-balance model. Simulates speed from power, and fits its own parameters to measured speed |
| `Rider` | MMP data. Fits the physiological models |
| `Optimiser` | Finds the best power profile for a vehicle and a fitted model |
| `OptimisationResult` | The optimised profile and the speed it achieves |

Units: time in s, power in W, speed in km/h, mass in kg.

## Usage

```python
import hpvopt as hv
```

### Load data

```python
run = hv.Run("ride.tcx", start=450, end=950)
run.plot()
```

`Run` reads `.tcx` and `.npz` files. `start` and `end` are seconds from the
start of the file, and the returned run is shifted to start at 0. Run attributes
are numpy arrays: `times`, `watts`, `speeds` (measured), `calc_speeds`
(simulated), and `distances` and `cadences` when read from a TCX file.

To build a run from arrays:

```python
run = hv.Run.from_arrays(times, watts, speeds=None)
```

Other useful calls:

```python
part = run.slice(100, 300)    # copy of a section
run.save("run.npz")
run.plot(other_run, series=["watts", "speeds"])
```

Samples should be evenly spaced. The simulation takes its time step from the
first two samples and resets the vehicle's energy at any larger gap.

### Define a vehicle

```python
vehicle = hv.Vehicle(rider_mass=80, bike_mass=55, cda=0.030, crr=0.008,
                     air_density=1.1, slope=-1/150, wind=1.0)
print(vehicle)
```

`slope` is a gradient in m/m, negative downhill. `wind` is in m/s, positive for
a tailwind. Efficiency, wheel mass, number of wheels and gravity have defaults
and can be set as keyword arguments.

### Simulate speed from power

```python
sim = vehicle.simulate(run)
run.plot(sim, series=[["watts", "speeds"], ["calc_speeds"]])
```

`simulate` returns a copy of the run with `calc_speeds` set. It has one fewer
sample than `times`, since each value is the speed after a step.

### Fit vehicle parameters to measured speed

```python
fitted = vehicle.fit([run1, run2])
fitted = vehicle.fit([run1, run2], free=("cda", "crr"))
```

`fit` returns a new `Vehicle` and leaves the original unchanged. It minimises
the RMSE between simulated and measured speed over all runs. By default CdA,
Crr and air density are free. Bounds can be changed with
`bounds={"cda": (0.01, 0.5)}`. Every run needs measured `speeds`.

A run can carry its own rider mass, which `fit` uses in place of the vehicle's:

```python
run = hv.Run("ride.tcx", rider_mass=75)
```

Fitting several parameters to one short run is poorly constrained. Use
several runs, or fix some parameters with `free`.

### Fit a rider's physiological models

```python
mmp_times = [1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 1800, 3600]
mmp_watts = [1288, 1278, 1268, 1218, 792, 558, 477, 296, 244, 242, 228, 220, 204]

rider = hv.Rider(mmp_times, mmp_watts).fit()
rider.models["biexp"].params
rider.fit_quality()
rider.plot_fits()
```

| Key in `rider.models` | Model | Parameters |
|---|---|---|
| `"cp"` | Critical power and W' | `CP`, `W_prime` |
| `"exp"` | Single-exponential power ceiling | `Pmax`, `tau`, `CP` |
| `"biexp"` | Bi-exponential power ceiling | `A1`, `tau1`, `A2`, `tau2`, `CP` |
| `"disproportionate"` | Disproportionate W' expenditure | shares the `"cp"` fit |

The `"cp"` model is fitted to the 60 to 1800 s part of the MMP data only.
The others use the full range.

### Optimise

```python
opt = hv.Optimiser(fitted, rider.models["biexp"], duration=600, dt=2)
result = opt.run(n_jobs=4)

print(result.speed)       # fastest 200 m average, km/h
result.plot()
result.save("profile.npz")
```

The profile starts at the model's CP and is searched in the range `bounds`
(default 0 to 2000 W). The model then limits the realised power. The objective
is the negative fastest 200 m average speed, plus small penalties on power
changes (`roughness_weighting`) and on power drawn late in the run
(`time_weighting`). Both penalties apply only up to the end of the fastest
segment, so ignore the profile after it, and the noise early in the run.

To load a saved profile:

```python
result = hv.OptimisationResult.load("profile.npz", fitted)
```

The file stores only the power profile. The speed is recomputed for the
vehicle you pass.

### Compare with a measured attempt

```python
attempt = hv.Run("attempt.tcx", start=100, end=600)
attempt_sim = fitted.simulate(attempt)

hv.plot(result.run, attempt_sim,
        series=[["watts"], ["watts", "calc_speeds"]],
        labels=[{"watts": "Optimised Power"},
                {"watts": "Measured Power", "calc_speeds": "Simulated Speed"}],
        savefig="overlay.pdf")
```

`savefig` writes both a PDF and a PNG. `hv.paper_style()` sets the matplotlib
style used for the paper figures. It changes global matplotlib settings, so it
is not applied on import.

## Adding a physiological model

Subclass `hpvopt.models.Model` and provide:

- `fit(t, p)`: fit the model to MMP durations `t` and powers `p`, set
`self.params` (which must include `"CP"`), and return `self`.
- `predict(t)`: power the fitted curve gives at duration `t`.
- `realise(power_profile, dt)`: return the power the rider can actually
produce when asked for `power_profile` at time step `dt`.

Add the class to `MODELS` in `hpvopt/models.py` to have `Rider` fit it.
Alternatively, assign an instance directly: `rider.models["mine"] =
MyModel().fit(mmp_times, mmp_watts)`.

## Licence

hpvopt is released under the [GNU General Public License v3.0](LICENSE).

## AI declaration

The original code and the method are the author's own. The object-oriented
refactor into the `hpvopt` package, the docstrings and this README were
drafted with the assistance of Claude Sonnet 5.5 (Anthropic). The author
reviewed and tested the result and is responsible for its content.
