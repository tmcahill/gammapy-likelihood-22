import numpy as np
from copy import deepcopy

from gammapy.irf import Background2D

"""
Creates a background IRF from one or many Gammapy `Observations`

This class also returns a map of counts per bin to be used in cost statistics dealing with weighted binning.
"""
class BackgroundModelEstimator:
    def __init__(self, energy, offset):
        self.counts = self._make_bkg2d(energy, offset, unit="")
        self.exposure = self._make_bkg2d(energy, offset, unit="s MeV sr")


    @staticmethod
    def _make_bkg2d(energy, offset, unit):
        shape = len(energy.center), len(offset.center)
        return Background2D(axes=[energy, offset], unit=unit)


    """
    Can stack multiple observations into the background model by calling this multiple times or on a collection of Observations
    """
    def run(self, observationsbkg):
        for obs in observationsbkg:
        #for table in obstable:
            self.fill_counts(obs)
            self.fill_exposure(obs)
    

    """
    Fills the counts per bin for the defined background IRF geometry
    """
    def fill_counts(self, obs):
        events = obs.events

        energy_bins = self.counts.axes["energy"].edges
        offset_bins = self.counts.axes["offset"].edges

        counts = np.histogram2d(
            x=events.energy.to("MeV"),
            y=events.offset.to("deg"),
            bins=(energy_bins, offset_bins),
        )[0]

        self.counts.data += counts


    """
    Calculate exposure for each bin the background IRF geometry
    """
    def fill_exposure(self, obs):
        axes = self.exposure.axes
        offset = axes["offset"].center
        time = obs.observation_time_duration
        exposure = 2 * np.pi * offset * time * axes.bin_volume()
        self.exposure.quantity += exposure


    """
    Calculate and return the background model and stats for all of the observations ran thus far
    """
    @property
    def background_rate(self):
        rate = deepcopy(self.counts)
        rate.quantity /= self.exposure.quantity
        # Return rate and stats per bin
        return rate, self.counts.quantity