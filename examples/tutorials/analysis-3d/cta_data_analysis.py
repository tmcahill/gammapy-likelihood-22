"""
Basic image exploration and fitting
===================================

Detect sources, produce a sky image and a spectrum using the reflected regions background estimation.

Introduction
------------

**This notebook shows an example how to make a sky image and spectrum
for simulated CTA data with Gammapy.**

The dataset we will use is three observation runs on the Galactic
center. This is a tiny (and thus quick to process and play with and
learn) subset of the simulated CTA dataset that was produced for the
first data challenge in August 2017.

"""


######################################################################
# Setup
# -----
# 
# As usual, we’ll start with some setup …
# 

# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
from gammapy.modeling import Fit
from gammapy.data import DataStore
from gammapy.datasets import (
    Datasets,
    FluxPointsDataset,
    SpectrumDataset,
    MapDataset,
)
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    SkyModel,
    GaussianSpatialModel,
)
from gammapy.maps import MapAxis, WcsGeom, RegionGeom
from gammapy.makers import (
    MapDatasetMaker,
    SafeMaskMaker,
    SpectrumDatasetMaker,
    ReflectedRegionsBackgroundMaker,
)
from gammapy.estimators import TSMapEstimator, FluxPointsEstimator
from gammapy.estimators.utils import find_peaks
from gammapy.visualization import plot_spectrum_datasets_off_regions

# Configure the logger, so that the spectral analysis
# isn't so chatty about what it's doing.
import logging

logging.basicConfig()
log = logging.getLogger("gammapy.spectrum")
log.setLevel(logging.ERROR)

######################################################################
# Check setup
# -----------
from gammapy.utils.check import check_tutorials_setup

check_tutorials_setup()


######################################################################
# Select observations
# -------------------
# 
# A Gammapy analysis usually starts by creating a
# `~gammapy.data.DataStore` and selecting observations.
# 
# This is shown in detail in the other notebook, here we just pick three
# observations near the galactic center.
# 

data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps")

# Just as a reminder: this is how to select observations
# from astropy.coordinates import SkyCoord
# table = data_store.obs_table
# pos_obs = SkyCoord(table['GLON_PNT'], table['GLAT_PNT'], frame='galactic', unit='deg')
# pos_target = SkyCoord(0, 0, frame='galactic', unit='deg')
# offset = pos_target.separation(pos_obs).deg
# mask = (1 < offset) & (offset < 2)
# table = table[mask]
# table.show_in_browser(jsviewer=True)

obs_id = [110380, 111140, 111159]
observations = data_store.get_observations(obs_id)

obs_cols = ["OBS_ID", "GLON_PNT", "GLAT_PNT", "LIVETIME"]
data_store.obs_table.select_obs_id(obs_id)[obs_cols]


######################################################################
# Make sky images
# ---------------
# 
# Define map geometry
# ~~~~~~~~~~~~~~~~~~~
# 
# Select the target position and define an ON region for the spectral
# analysis
# 

axis = MapAxis.from_edges(
    np.logspace(-1.0, 1.0, 10), unit="TeV", name="energy", interp="log"
)
geom = WcsGeom.create(
    skydir=(0, 0), npix=(500, 400), binsz=0.02, frame="galactic", axes=[axis]
)
geom


######################################################################
# Compute images
# ~~~~~~~~~~~~~~
# 

# %%time
stacked = MapDataset.create(geom=geom)
stacked.edisp = None
maker = MapDatasetMaker(selection=["counts", "background", "exposure", "psf"])
maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=2.5 * u.deg)

for obs in observations:
    cutout = stacked.cutout(obs.pointing_radec, width="5 deg")
    dataset = maker.run(cutout, obs)
    dataset = maker_safe_mask.run(dataset, obs)
    stacked.stack(dataset)

# The maps are cubes, with an energy axis.
# Let's also make some images:
dataset_image = stacked.to_image()


######################################################################
# Show images
# ~~~~~~~~~~~
# 
# Let’s have a quick look at the images we computed …
# 

dataset_image.counts.smooth(2).plot(vmax=5);

dataset_image.background.plot(vmax=5);

dataset_image.excess.smooth(3).plot(vmax=2);


######################################################################
# Source Detection
# ----------------
# 
# Use the class `~gammapy.estimators.TSMapEstimator` and function
# `gammapy.estimators.utils.find_peaks` to detect sources on the images.
# We search for 0.1 deg sigma gaussian sources in the dataset.
# 

spatial_model = GaussianSpatialModel(sigma="0.05 deg")
spectral_model = PowerLawSpectralModel(index=2)
model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)

ts_image_estimator = TSMapEstimator(
    model,
    kernel_width="0.5 deg",
    selection_optional=[],
    downsampling_factor=2,
    sum_over_energy_groups=False,
    energy_edges=[0.1, 10] * u.TeV,
)

# %%time
images_ts = ts_image_estimator.run(stacked)

sources = find_peaks(
    images_ts["sqrt_ts"],
    threshold=5,
    min_distance="0.2 deg",
)
sources

source_pos = SkyCoord(sources["ra"], sources["dec"])
source_pos

# Plot sources on top of significance sky image
images_ts["sqrt_ts"].plot(add_cbar=True)

plt.gca().scatter(
    source_pos.ra.deg,
    source_pos.dec.deg,
    transform=plt.gca().get_transform("icrs"),
    color="none",
    edgecolor="white",
    marker="o",
    s=200,
    lw=1.5,
);


######################################################################
# Spatial analysis
# ----------------
# 
# See other notebooks for how to run a 3D cube or 2D image based analysis.
# 


######################################################################
# Spectrum
# --------
# 
# We’ll run a spectral analysis using the classical reflected regions
# background estimation method, and using the on-off (often called WSTAT)
# likelihood function.
# 

target_position = SkyCoord(0, 0, unit="deg", frame="galactic")
on_radius = 0.2 * u.deg
on_region = CircleSkyRegion(center=target_position, radius=on_radius)

exclusion_mask = ~geom.to_image().region_mask([on_region])
exclusion_mask.plot();

energy_axis = MapAxis.from_energy_bounds(
    0.1, 40, 40, unit="TeV", name="energy"
)
energy_axis_true = MapAxis.from_energy_bounds(
    0.05, 100, 200, unit="TeV", name="energy_true"
)

geom = RegionGeom.create(region=on_region, axes=[energy_axis])
dataset_empty = SpectrumDataset.create(
    geom=geom, energy_axis_true=energy_axis_true
)

dataset_maker = SpectrumDatasetMaker(
    containment_correction=False, selection=["counts", "exposure", "edisp"]
)
bkg_maker = ReflectedRegionsBackgroundMaker(exclusion_mask=exclusion_mask)
safe_mask_masker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)

# %%time
datasets = Datasets()

for observation in observations:
    dataset = dataset_maker.run(
        dataset_empty.copy(name=f"obs-{observation.obs_id}"), observation
    )
    dataset_on_off = bkg_maker.run(dataset, observation)
    dataset_on_off = safe_mask_masker.run(dataset_on_off, observation)
    datasets.append(dataset_on_off)

plt.figure(figsize=(8, 8))
ax = dataset_image.counts.smooth("0.03 deg").plot(vmax=8)

on_region.to_pixel(ax.wcs).plot(ax=ax, edgecolor="white")
plot_spectrum_datasets_off_regions(datasets, ax=ax)


######################################################################
# Model fit
# ~~~~~~~~~
# 
# The next step is to fit a spectral model, using all data (i.e. a
# “global” fit, using all energies).
# 

# %%time
spectral_model = PowerLawSpectralModel(
    index=2, amplitude=1e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
)

model = SkyModel(spectral_model=spectral_model, name="source-gc")

datasets.models = model

fit = Fit()
result = fit.run(datasets=datasets)
print(result)


######################################################################
# Spectral points
# ~~~~~~~~~~~~~~~
# 
# Finally, let’s compute spectral points. The method used is to first
# choose an energy binning, and then to do a 1-dim likelihood fit /
# profile to compute the flux and flux error.
# 

# Flux points are computed on stacked observation
stacked_dataset = datasets.stack_reduce(name="stacked")

print(stacked_dataset)

energy_edges = MapAxis.from_energy_bounds("1 TeV", "30 TeV", nbin=5).edges

stacked_dataset.models = model

fpe = FluxPointsEstimator(energy_edges=energy_edges, source="source-gc")
flux_points = fpe.run(datasets=[stacked_dataset])
flux_points.to_table(sed_type="dnde", formatted=True)


######################################################################
# Plot
# ~~~~
# 
# Let’s plot the spectral model and points. You could do it directly, but
# for convenience we bundle the model and the flux points in a
# `FluxPointDataset`:
# 

flux_points_dataset = FluxPointsDataset(data=flux_points, models=model)

flux_points_dataset.plot_fit();


######################################################################
# Exercises
# ---------
# 
# -  Re-run the analysis above, varying some analysis parameters, e.g.
# 
#    -  Select a few other observations
#    -  Change the energy band for the map
#    -  Change the spectral model for the fit
#    -  Change the energy binning for the spectral points
# 
# -  Change the target. Make a sky image and spectrum for your favourite
#    source.
# 
#    -  If you don’t know any, the Crab nebula is the “hello world!”
#       analysis of gamma-ray astronomy.
# 

# print('hello world')
# SkyCoord.from_name('crab')


######################################################################
# What next?
# ----------
# 
# -  This notebook showed an example of a first CTA analysis with Gammapy,
#    using simulated 1DC data.
# -  Let us know if you have any question or issues!
# 