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

    index_denom = log(energy_min / energy_max)
    rate_ratio = vals_energy_min / vals_energy_max
    index_numer = log(rate_ratio)
    with np.errstate(invalid="ignore", divide="ignore"):
        index = -index_numer / index_denom

    index[np.isnan(index)] = np.inf

    if weights is None:
        return pl.evaluate_integral(
            energy_min=energy_min,
            energy_max=energy_max,
            index=index,
            reference=energy_min,
            amplitude=vals_energy_min,
        )

    errors = weights["errors"].reshape(y.shape)
    vals_error_min, vals_error_max = errors[:-1], errors[1:]
    # print(vals_error_min[0][99])
    # print("VALS_ERROR_MIN:", vals_error_min.unit)
    # print("VALS_energy_MIN:", vals_energy_min.unit)
    rate_ratio_error = rate_ratio * np.sqrt(
        np.square(vals_error_min / vals_energy_min) + (np.square(vals_error_max / vals_energy_max))
    )
    # print("RATE_RATIO_ERROR:", rate_ratio_error.unit)
    # print("RATE_RATIO:", rate_ratio.unit)
    # Error of ln(x) is error(x)/x
    index_numer_error = rate_ratio_error / rate_ratio
    with np.errstate(invalid="ignore", divide="ignore"):
        # d/dx where x = log(rate_ratio)
        index_error = np.abs(1/index_denom) * index_numer_error

    return pl.evaluate_integral_with_rate_errors(
        energy_min=energy_min,
        energy_max=energy_max,
        index=index,
        reference=energy_min,
        amplitude=vals_energy_min,
        index_error=index_error,
        amp_error=vals_error_min,
        r_0_err_term=np.square(vals_error_min / vals_energy_min), 
        r_1_err_term=np.square(vals_error_max / vals_energy_max), 
        Q= -index_denom - index_numer,
    )

    # print("NP EQUAL:", np.array_equal(pl1, pl2))
    # print(pl2[3][99])
    # print(pl2_errors[3][99])
    # print("-------------")
    # print(pl2[6][99])
    # print(pl2_errors[6][99])

    """ Working, but very slow """
    # if weights is not None:
    #     #pl2 = np.empty(pl1.shape)
    #     pl2 = pl1.copy()
    #     pl2[True] = np.nan
    #     pl2 = pl2.astype(astropy.units.Quantity)
    #     print("SHAPES: ", pl1.shape, pl2.shape, energy_min.shape)
    #     print(energy_min)
    #     errors = weights["errors"].reshape(y.shape)
    #     errors = errors[:-1]
    #     e_unit = energy_min.unit
    #     energy_min = np.broadcast_to(energy_min, vals_energy_min.shape)
    #     energy_max = np.broadcast_to(energy_max, vals_energy_max.shape)
    #     #print(energy_min)
    #     powerlaw = pl()

    #     powerlaw.reference.unit = e_unit
    #     powerlaw.amplitude.unit = vals_energy_min.flat[0].unit
    #     import timeit
    #     start_time = timeit.default_timer()
    #     for arr_idx in np.ndindex(vals_energy_min.shape):
    #         #print(index[arr_idx], energy_min[arr_idx], vals_energy_min[arr_idx])
    #         #pl.amplitude.unit = vals_energy_min[arr_idx.unit]
    #         """
    #         powerlaw = pl(
    #             index=index[arr_idx],
    #             reference=energy_min[arr_idx] * e_unit,
    #             amplitude=astropy.units.Quantity(
    #                 vals_energy_min[arr_idx].value, 
    #                 #TODO Unit hardcoding bad!
    #                 unit="1 / (cm2 s MeV)"
    #             )
    #             #amplitude = vals_energy_min[arr_idx]
    #         )
    #         powerlaw.amplitude.unit = vals_energy_min[arr_idx].unit
    #         """
    #         powerlaw.index.value = index[arr_idx]
    #         powerlaw.reference.value = energy_min[arr_idx]
    #         powerlaw.amplitude.value = vals_energy_min[arr_idx].value
    #         powerlaw.amplitude.error = errors[arr_idx]

    #         # powerlaw = pl

    #         # powerlaw.index = index[arr_idx]
    #         # powerlaw.reference = energy_min[arr_idx]
    #         # powerlaw.amplitude = vals_energy_min[arr_idx]

    #         # print("arr_idx: ", arr_idx)
    #         # print(pl2[arr_idx])
            
    #         """pl2[arr_idx] = powerlaw.integral(        
    #             energy_min=energy_min[arr_idx] * e_unit,
    #             energy_max=energy_max[arr_idx] * e_unit,
    #         )"""
    #         ele = powerlaw.integral_error(
    #             energy_min=energy_min[arr_idx] * e_unit,
    #             energy_max=energy_max[arr_idx] * e_unit,
    #         )
    #         print(ele)
    #         pl2[arr_idx] = ele
    #         # print(arr_idx)
    #         # print(pl1[arr_idx])
    #         # print(ele)  

    #     print("NP EQUAL:", np.array_equal(pl1, pl2))
    #     #print(pl2)
    #     print("RUNTIME: ", timeit.default_timer() - start_time)
