# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import astropy
from .interpolation import LogScale

__all__ = ["trapz_loglog"]


def trapz_loglog(y, x, axis=-1, weights=None):
    """Integrate using the composite trapezoidal rule in log-log space.

    Integrate `y` (`x`) along given axis in loglog space.

    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        Independent variable to integrate over.
    axis : int, optional
        Specify the axis.

    Returns
    -------
    trapz : float
        Definite integral as approximated by trapezoidal rule in loglog space.
    """
    from gammapy.modeling.models import PowerLawSpectralModel as pl

    # see https://stackoverflow.com/a/56840428
    x, y = np.moveaxis(x, axis, 0), np.moveaxis(y, axis, 0)

    energy_min, energy_max = x[:-1], x[1:]
    vals_energy_min, vals_energy_max = y[:-1], y[1:]

    # log scale has the built-in zero clipping
    log = LogScale()

    with np.errstate(invalid="ignore", divide="ignore"):
        index = -log(vals_energy_min / vals_energy_max) / log(energy_min / energy_max)

    index[np.isnan(index)] = np.inf

    pl1 = pl.evaluate_integral(
        energy_min=energy_min,
        energy_max=energy_max,
        index=index,
        reference=energy_min,
        amplitude=vals_energy_min,
    )

    if weights is not None:
        #pl2 = np.empty(pl1.shape)
        pl2 = pl1.copy()
        pl2[True] = np.nan
        pl2 = pl2.astype(astropy.units.Quantity)
        print("SHAPES: ", pl1.shape, pl2.shape, energy_min.shape)
        print(energy_min)
        errors = weights["errors"].reshape(y.shape)
        errors = errors[:-1]
        e_unit = energy_min.unit
        energy_min = np.broadcast_to(energy_min, vals_energy_min.shape)
        energy_max = np.broadcast_to(energy_max, vals_energy_max.shape)
        #print(energy_min)
        powerlaw = pl()

        powerlaw.reference.unit = e_unit
        powerlaw.amplitude.unit = vals_energy_min.flat[0].unit
        import timeit
        start_time = timeit.default_timer()
        for arr_idx in np.ndindex(vals_energy_min.shape):
            #print(index[arr_idx], energy_min[arr_idx], vals_energy_min[arr_idx])
            #pl.amplitude.unit = vals_energy_min[arr_idx.unit]
            """
            powerlaw = pl(
                index=index[arr_idx],
                reference=energy_min[arr_idx] * e_unit,
                amplitude=astropy.units.Quantity(
                    vals_energy_min[arr_idx].value, 
                    #TODO Unit hardcoding bad!
                    unit="1 / (cm2 s MeV)"
                )
                #amplitude = vals_energy_min[arr_idx]
            )
            powerlaw.amplitude.unit = vals_energy_min[arr_idx].unit
            """
            powerlaw.index.value = index[arr_idx]
            powerlaw.reference.value = energy_min[arr_idx]
            powerlaw.amplitude.value = vals_energy_min[arr_idx].value
            powerlaw.amplitude.error = errors[arr_idx]

            # powerlaw = pl

            # powerlaw.index = index[arr_idx]
            # powerlaw.reference = energy_min[arr_idx]
            # powerlaw.amplitude = vals_energy_min[arr_idx]

            # print("arr_idx: ", arr_idx)
            # print(pl2[arr_idx])
            
            """pl2[arr_idx] = powerlaw.integral(        
                energy_min=energy_min[arr_idx] * e_unit,
                energy_max=energy_max[arr_idx] * e_unit,
            )"""
            ele = powerlaw.integral_error(
                energy_min=energy_min[arr_idx] * e_unit,
                energy_max=energy_max[arr_idx] * e_unit,
            )
            print(ele)
            pl2[arr_idx] = ele
            # print(arr_idx)
            # print(pl1[arr_idx])
            # print(ele)  

        print("NP EQUAL:", np.array_equal(pl1, pl2))
        #print(pl2)
        print("RUNTIME: ", timeit.default_timer() - start_time)

    return pl1