# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Interpolation utilities"""
from itertools import compress
import itertools 
import numpy as np
import scipy.interpolate
from astropy import units as u

__all__ = [
    "interpolate_profile",
    "interpolation_scale",
    "ScaledRegularGridInterpolator",
]

INTERPOLATION_ORDER = {None: 0, "nearest": 0, "linear": 1, "quadratic": 2, "cubic": 3}


class ScaledRegularGridInterpolator:
    """Thin wrapper around `scipy.interpolate.RegularGridInterpolator`.

    The values are scaled before the interpolation and back-scaled after the
    interpolation.

    Dimensions of length 1 are ignored in the interpolation of the data.

    Parameters
    ----------
    points : tuple of `~numpy.ndarray` or `~astropy.units.Quantity`
        Tuple of points passed to `RegularGridInterpolator`.
    values : `~numpy.ndarray`
        Values passed to `RegularGridInterpolator`.
    points_scale : tuple of str
        Interpolation scale used for the points.
    values_scale : {'lin', 'log', 'sqrt'}
        Interpolation scaling applied to values. If the values vary over many magnitudes
        a 'log' scaling is recommended.
    axis : int or None
        Axis along which to interpolate.
    method : {"linear", "nearest"}
        Default interpolation method. Can be overwritten when calling the
        `ScaledRegularGridInterpolator`.
    **kwargs : dict
        Keyword arguments passed to `RegularGridInterpolator`.
    """

    def __init__(
        self,
        points,
        values,
        points_scale=None,
        values_scale="lin",
        extrapolate=True,
        axis=None,
        **kwargs,
    ):

        if points_scale is None:
            points_scale = ["lin"] * len(points)

        self.scale_points = [interpolation_scale(scale) for scale in points_scale]
        self.scale = interpolation_scale(values_scale)
        self.axis = axis

        self._include_dimensions = [len(p) > 1 for p in points]

        values_scaled = self.scale(values)
        points_scaled = self._scale_points(points=points)

        self._values_scaled = values_scaled
        self._points_scaled = points_scaled
        self._kwargs = kwargs

        if extrapolate:
            kwargs.setdefault("bounds_error", False)
            kwargs.setdefault("fill_value", None)

        method = kwargs.get("method", None)

        if not np.any(self._include_dimensions):
            if method != "nearest":
                raise ValueError(
                    "Interpolating scalar values requires using "
                    "method='nearest' explicitely."
                )

        if np.any(self._include_dimensions):
            values_scaled = np.squeeze(values_scaled)

        if axis is None:
            self._interpolate = scipy.interpolate.RegularGridInterpolator(
                points=points_scaled, values=values_scaled, **kwargs
            )
        else:
            self._interpolate = scipy.interpolate.interp1d(
                points_scaled[0], values_scaled, axis=axis
            )

    def _scale_points(self, points):
        points_scaled = [scale(p) for p, scale in zip(points, self.scale_points)]

        if np.any(self._include_dimensions):
            points_scaled = compress(points_scaled, self._include_dimensions)

        return tuple(points_scaled)

    def __call__(self, points, method=None, clip=True, get_weights=False, **kwargs):
        """Interpolate data points.

        Parameters
        ----------
        points : tuple of `~numpy.ndarray` or `~astropy.units.Quantity`
            Tuple of coordinate arrays of the form (x_1, x_2, x_3, ...). Arrays are
            broadcasted internally.
        method : {None, "linear", "nearest"}
            Linear or nearest neighbour interpolation. None will choose the default
            defined on init.
        clip : bool
            Clip values at zero after interpolation.
        """
        points = self._scale_points(points=points)

        if self.axis is None:
            points = np.broadcast_arrays(*points)
            points_interp = np.stack([_.flat for _ in points]).T
            if get_weights:
                weighted_interpolator = RegularGridInterpolatorWithWeights(points=self._points_scaled, values=self._values_scaled, **self._kwargs)
                values, weights = weighted_interpolator(points_interp, method, get_weights=get_weights, **kwargs)
                #print(weights["errors"])
                values = self.scale.inverse(values.reshape(points[0].shape))
                # TODO: handle appropriately for different scales?
                #print("val scale:", self.scale)
                
                weights["errors"] = self.scale.inverse(weights["errors"].reshape(points[0].shape))
                #print(weights["errors"])
            else:
                values = self._interpolate(points_interp, method, **kwargs)
                values = self.scale.inverse(values.reshape(points[0].shape))

        else:
            values = self._interpolate(points[0])
            values = self.scale.inverse(values)

        if clip:
            values = np.clip(values, 0, np.inf)

        if get_weights:
             # Reshape weights arrays to match new geom.
            weights["weights"] = np.array(
                weights["weights"],
                # TODO: not have this hardcoded to 4 values
                dtype=[('a', np.float64),('b', np.float64),('c', np.float64),('d', np.float64)]
            ).reshape(values.shape)

            weights["indices"] = np.array(
                # TODO: not have this hardcoded to 4 values
                weights["indices"], 
                dtype=[('a', np.ndarray),('b', np.ndarray),('c', np.ndarray),('d', np.ndarray)]
            ).reshape(values.shape)
            
            return values, weights

        return values


def interpolation_scale(scale="lin"):
    """Interpolation scaling.

    Parameters
    ----------
    scale : {"lin", "log", "sqrt"}
        Choose interpolation scaling.
    """
    if scale in ["lin", "linear"]:
        return LinearScale()
    elif scale == "log":
        return LogScale()
    elif scale == "sqrt":
        return SqrtScale()
    elif scale == "stat-profile":
        return StatProfileScale()
    elif isinstance(scale, InterpolationScale):
        return scale
    else:
        raise ValueError(f"Not a valid value scaling mode: '{scale}'.")


class InterpolationScale:
    """Interpolation scale base class."""

    def __call__(self, values):
        if hasattr(self, "_unit"):
            values = u.Quantity(values, copy=False).to_value(self._unit)
        else:
            if isinstance(values, u.Quantity):
                self._unit = values.unit
                values = values.value
        return self._scale(values)

    def inverse(self, values):
        values = self._inverse(values)
        if hasattr(self, "_unit"):
            return u.Quantity(values, self._unit, copy=False)
        else:
            return values


class LogScale(InterpolationScale):
    """Logarithmic scaling"""

    tiny = np.finfo(np.float32).tiny

    def _scale(self, values):
        values = np.clip(values, self.tiny, np.inf)
        return np.log(values)

    @classmethod
    def _inverse(cls, values):
        output = np.exp(values)
        return np.where(abs(output) - cls.tiny <= cls.tiny, 0, output)


class SqrtScale(InterpolationScale):
    """Sqrt scaling"""

    @staticmethod
    def _scale(values):
        sign = np.sign(values)
        return sign * np.sqrt(sign * values)

    @classmethod
    def _inverse(cls, values):
        return np.power(values, 2)


class StatProfileScale(InterpolationScale):
    """Sqrt scaling"""

    def __init__(self, axis=0):
        self.axis = axis

    def _scale(self, values):
        values = np.sign(np.gradient(values, axis=self.axis)) * values
        sign = np.sign(values)
        return sign * np.sqrt(sign * values)

    @classmethod
    def _inverse(cls, values):
        return np.power(values, 2)


class LinearScale(InterpolationScale):
    """Linear scaling"""

    @staticmethod
    def _scale(values):
        return values

    @classmethod
    def _inverse(cls, values):
        return values


def interpolate_profile(x, y, interp_scale="sqrt"):
    """Helper function to interpolate one-dimensional profiles.

    Parameters
    ----------
    x : `~numpy.ndarray`
        Array of x values
    y : `~numpy.ndarray`
        Array of y values
    interp_scale : {"sqrt", "lin"}
        Interpolation scale applied to the profile. If the profile is
        of parabolic shape, a "sqrt" scaling is recommended. In other cases or
        for fine sampled profiles a "lin" can also be used.

    Returns
    -------
    interp : `ScaledRegularGridInterpolator`
        Interpolator
    """
    sign = np.sign(np.gradient(y))
    return ScaledRegularGridInterpolator(
        points=(x,), values=sign * y, values_scale=interp_scale
    )


"""
This class also returns the interpolation weights per bin to be used in cost statistics dealing with weighted binning.
"""
class RegularGridInterpolatorWithWeights(scipy.interpolate.RegularGridInterpolator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, xi, method=None, bkg_stats=None, get_weights=False):
        from scipy.interpolate.interpnd import _ndim_coords_from_arrays
        """
        Interpolation at coordinates

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at

        method : str
            The method of interpolation to perform. Supported are "linear" and
            "nearest".

        """
        method = self.method if method is None else method
        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)

        ndim = len(self.grid)
        xi = _ndim_coords_from_arrays(xi, ndim=ndim)
        if xi.shape[-1] != len(self.grid):
            raise ValueError("The requested sample points xi have dimension "
                             "%d, but this RegularGridInterpolator has "
                             "dimension %d" % (xi.shape[1], ndim))

        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])

        if self.bounds_error:
            for i, p in enumerate(xi.T):
                if not np.logical_and(np.all(self.grid[i][0] <= p),
                                      np.all(p <= self.grid[i][-1])):
                    raise ValueError("One of the requested xi is out of bounds "
                                     "in dimension %d" % i)

        indices, norm_distances, out_of_bounds = self._find_indices(xi.T)
        if method == "linear":
            if get_weights:
                result, weights = self._evaluate_linear(indices,
                                                        norm_distances,
                                                        out_of_bounds,
                                                        bkg_stats=bkg_stats,
                                                        get_weights=get_weights)
            else:
                result = self._evaluate_linear(indices,
                                               norm_distances,
                                               out_of_bounds)
        elif method == "nearest":
            result = self._evaluate_nearest(indices,
                                            norm_distances,
                                            out_of_bounds)
        if not self.bounds_error and self.fill_value is not None:
            result[out_of_bounds] = self.fill_value
            if get_weights:
                weights["weights"][out_of_bounds] = None
                weights["indices"][out_of_bounds] = None
                #weights["errors"] = np.reshape(weights["errors"], (xi_shape[:-1] + self.values.shape[ndim:]))

        if get_weights:
            return result.reshape(xi_shape[:-1] + self.values.shape[ndim:]), weights
        else:
            return result.reshape(xi_shape[:-1] + self.values.shape[ndim:])

    """
    Debug prints left in for now
    Needs optimized
    """
    def _evaluate_linear(self, indices, norm_distances, out_of_bounds, bkg_stats=None, get_weights=False):
        with np.printoptions(threshold=1000, edgeitems=50):
            vslice = (slice(None),) + (None,)*(self.values.ndim - len(indices))

            # find relevant values
            # each i and i+1 represents a edge
            edges = itertools.product(*[[i, i + 1] for i in indices])
            values = 0.
            errors = 0.
            unweighted_errors = 0
            weights = []
            weights_indices = []
            bkg_errors = bkg_stats

            # 4x, once per bin edges
            for edge_indices in edges:
                weight = 1.
                # print("OUTER LOOP")
                # 2x, once per dimension index
                inner_weights_arr = []
                for ei, i, yi in zip(edge_indices, indices, norm_distances):
                    # print("INNER LOOP")

                    # print(edge_indices)
                    # print(indices) 
                    # print(norm_distances)

                    # print("ei")
                    # print(ei)
                    # print("i")
                    # print(i)
                    # print("yi")
                    # print(yi)

                    weight *= np.where(ei == i, 1 - yi, yi)

                    # print("weight")
                    # print(weight)

                    if get_weights:
                        inner_weights_arr.append(weight.copy())

                # print("OUTER LOOP PT2")
                # print(self.values[edge_indices])
                # print(weight[vslice])
                values += np.asarray(self.values[edge_indices]) * weight[vslice]
                unweighted_errors += np.asarray(bkg_errors[edge_indices])
                errors += (np.asarray(bkg_errors[edge_indices]) * weight[vslice]) ** 2

                """Place weights in array"""
                if get_weights:
                    # errors.append(np.column_stack(bkg_stats[edge_indices]))
                    weights.append(weight[vslice])
                    weights_indices.append(np.column_stack(edge_indices))

            # List/zip right now is for formatting the dtype easily
            # print('bkg_stats')
            # print(bkg_stats)
            # print("self.values")
            # print(self.values)
            # print(errors)
            # print(unweighted_errors)
            # print(weights_indices)

            if get_weights:
                return values, {
                        "weights": list(zip(*weights)), 
                        "indices": list(zip(*weights_indices)), 
                        "values": self.values,
                        "interp_vals": values,
                        "errors": np.sqrt(errors),
                        "unweighted_errors": unweighted_errors
                    }
            else:
                return values