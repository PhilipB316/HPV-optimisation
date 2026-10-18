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

import copy
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from scipy.ndimage import uniform_filter1d
import numpy as np

from . import plotting

NS = {
    'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2',
    'ns3': 'http://www.garmin.com/xmlschemas/ActivityExtension/v2',
}

SERIES = ('times', 'distances', 'cadences', 'watts', 'speeds', 'calc_speeds')

def _text(elem, path, ns, cast=str):
    """Find a sub-element and cast its text, or return None if missing/empty."""
    found = elem.find(path, ns)
    if found is None or found.text is None:
        return None
    return cast(found.text)


def get_tcx_data(file_path):
    """
    Parses a TCX file to extract time, distance, cadence, and power data.
    """
    root = ET.parse(file_path).getroot()
    data = dict(times=[], distances=[], cadences=[], watts=[], speeds=[])

    for tp in root.findall('.//tcx:Trackpoint', NS):
        data['times'].append(_text(tp, 'tcx:Time', NS))
        data['distances'].append(_text(tp, 'tcx:DistanceMeters', NS, float) or 0)
        data['cadences'].append(_text(tp, 'tcx:Cadence', NS, int) or 0)

        watts_val = speed_val = None
        tpx = tp.find('tcx:Extensions/ns3:TPX', NS)
        if tpx is not None:
            watts_val = _text(tpx, 'ns3:Watts', NS, int)
            speed_val = _text(tpx, 'ns3:Speed', NS, float)
            if speed_val is not None:
                speed_val *= 3.6  # m/s -> km/h

        data['watts'].append(watts_val or 0)
        data['speeds'].append(speed_val or 0)

    if data['times'] and data['times'][0] is not None:
        t0 = datetime.fromisoformat(data['times'][0].replace('Z', '+00:00'))
        data['times'] = [
            (datetime.fromisoformat(t.replace('Z', '+00:00')) - t0).total_seconds()
            if t is not None else 0
            for t in data['times']
        ]
    return data


class Run:
    """Power and speed against time for one ride.

    Load from a .tcx or .npz file, or build from arrays with `Run.from_arrays`.
    Attributes are numpy arrays, or None if not available:
        times: seconds from the start of the run.
        watts: power in W.
        speeds: measured speed in km/h.
        calc_speeds: simulated speed in km/h, set by `Vehicle.simulate`.
        distances, cadences: from the TCX file only.
    """

    def __init__(self, path=None, start=None, end=None, label=None,
                 rider_mass=None, smooth_window=None):
        """Load a run from a file.

        Args:
            path: .tcx or .npz file. If None, an empty Run is created.
            start: first time to keep, in s from the start of the file.
            end: last time to keep, in s from the start of the file.
            label: name used in plot legends. Defaults to the file name.
            rider_mass: rider mass in kg for this run. If set, `Vehicle.fit`
                uses it instead of the vehicle's own rider_mass.

        Raises:
            ValueError: if the file type is not .tcx or .npz, or no samples
                fall between `start` and `end`.
        """
        for key in SERIES:
            setattr(self, key, None)
        self.label = label or (os.path.basename(path) if path else "Run")
        self.rider_mass = rider_mass
        if path is None:
            return

        ext = os.path.splitext(path)[1].lower()
        if ext == '.tcx':
            data = get_tcx_data(path)
        elif ext == '.npz':
            data = dict(np.load(path))
        else:
            raise ValueError(f"Unsupported file type '{ext}'. Expected .tcx or .npz")
        for key, values in data.items():
            setattr(self, key, np.array(values))

        if smooth_window and self.speeds is not None:
            self.speeds = uniform_filter1d(self.speeds.astype(float), smooth_window)

        if start is not None or end is not None:
            self.__dict__.update(self.slice(start, end).__dict__)

    @classmethod
    def from_arrays(cls, times, watts, speeds=None, label="Run"):
        """Build a Run from arrays.

        Args:
            times: sample times in s.
            watts: power in W, same length as `times`.
            speeds: measured speed in km/h, optional.
            label: name used in plot legends.
        """
        run = cls(label=label)
        run.times = np.asarray(times, float)
        run.watts = np.asarray(watts, float)
        if speeds is not None:
            run.speeds = np.asarray(speeds, float)
        return run

    def slice(self, start=None, end=None):
        """Return a copy containing only the samples between `start` and `end`.

        Times in the copy are shifted to start at 0. The original is unchanged.

        Args:
            start: first time to keep, in s. Defaults to the start of the run.
            end: last time to keep, in s. Defaults to the end of the run.
        """
        start = self.times[0] if start is None else start
        end = self.times[-1] if end is None else end

        indices = [i for i, t in enumerate(self.times) if start <= t <= end]
        if not indices:
            raise ValueError(
                f"No valid samples between {start} and {end} s "
                f"Run covers {self.times[0]} to {self.times[-1]} s")

        sliced = copy.copy(self)
        for key in SERIES:
            values = getattr(self, key)
            if values is not None:
                setattr(sliced, key, np.array([values[i] for i in indices if i < len(values)]))

        sliced.times = sliced.times - sliced.times[0]
        return sliced

    def save(self, path):
        """Save all available series to a .npz file that `Run(path)` can load."""

        np.savez(path, **{k: getattr(self, k) for k in SERIES \
            if getattr(self, k) is not None})

    def plot(self, *others, **kwargs):
        """Plot this run, optionally overlaid with other runs.

        Args:
            *others: further Run objects to overlay.
            **kwargs: passed to `hpvopt.plot` (series, labels, show, savefig).

        Example:
            run.plot(sim, series=["watts", "speeds", "calc_speeds"])
        """
        return plotting.plot(self, *others, **kwargs)

