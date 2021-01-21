# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy import units as u
from gammapy.maps import MapAxes, MapAxis
from gammapy.utils.array import array_stats_str
from ..core import IRF


class PSF(IRF):
    """PSF base class"""

    def normalize(self):
        """Normalize PSF to integrate to unity"""
        rad_max = self.axes["rad"].edges.max()
        self.data /= self.containment(rad=rad_max)

    def containment(self, rad, **kwargs):
        """Containment tof the PSF at given axes coordinates

        Parameters
        ----------
        rad : `~astropy.units.Quantity`
            Rad value
        **kwargs : dict
            Other coordinates

        Returns
        -------
        containment : `~numpy.ndarray`
            Containment
        """
        containment = self.integral(axis_name="rad", rad=rad, **kwargs)
        return np.clip(containment.to(""), 0, 1)

    def containment_radius(self, fraction, factor=20, **kwargs):
        """Containment radius at given axes coordinates

        Parameters
        ----------
        fraction : float or `~numpy.ndarray`
            Containment fraction
        factor : int
            Up-sampling factor of the rad axis, determines the precision of the
            computed containment radius.
        **kwargs : dict
            Other coordinates

        Returns
        -------
        radius : `~astropy.coordinates.Angle`
            Containment radius
        """
        # TODO: this uses a lot of numpy broadcasting tricks, maybe simplify...
        from gammapy.datasets.map import RAD_AXIS_DEFAULT
        output = np.broadcast(*kwargs.values(), fraction)

        try:
            rad_axis = self.axes["rad"]
        except KeyError:
            rad_axis = RAD_AXIS_DEFAULT

        # upsample for better precision
        rad = rad_axis.upsample(factor=factor).center

        axis = tuple(range(output.ndim))
        rad = np.expand_dims(rad, axis=axis).T
        containment = self.containment(rad=rad, **kwargs)

        fraction_idx = np.argmin(np.abs(containment - fraction), axis=0)
        return rad[fraction_idx].reshape(output.shape)

    def to_energy_dependent_table_psf(self, offset, rad=None):
        """Convert to energy-dependent table PSF.

        Parameters
        ----------
        offset : `~astropy.coordinates.Angle`
            Offset in the field of view. Default theta = 0 deg
        rad : `~astropy.coordinates.Angle`
            Offset from PSF center used for evaluating the PSF on a grid.
            Default offset = [0, 0.005, ..., 1.495, 1.5] deg.

        Returns
        -------
        table_psf : `~gammapy.irf.EnergyDependentTablePSF`
            Energy-dependent PSF
        """
        from gammapy.irf import EnergyDependentTablePSF
        from gammapy.datasets.map import RAD_AXIS_DEFAULT

        energy_axis_true = self.axes["energy_true"]

        if rad is None:
            rad_axis = RAD_AXIS_DEFAULT
        else:
            rad_axis = MapAxis.from_edges(rad, name="rad")

        axes = MapAxes([energy_axis_true, rad_axis])
        data = self.evaluate(**axes.get_coord(), offset=offset)
        return EnergyDependentTablePSF(
            axes=axes,
            data=data.value,
            unit=data.unit
        )

    def info(
        self,
        fraction=[0.68, 0.95],
        energy_true=[[1.0], [10.0]] * u.TeV,
        offset=0*u.deg,
    ):
        """
        Print PSF summary info.

        The containment radius for given fraction, energies and thetas is
        computed and printed on the command line.

        Parameters
        ----------
        fraction : list
            Containment fraction to compute containment radius for.
        energy_true : `~astropy.units.u.Quantity`
            Energies to compute containment radius for.
        offset : `~astropy.units.u.Quantity`
            Offset to compute containment radius for.

        Returns
        -------
        ss : string
            Formatted string containing the summary info.
        """
        info = "\nSummary PSF info\n"
        info += "----------------\n"
        info += array_stats_str(self.axes["offset"].center.to("deg"), "Theta")
        info += array_stats_str(self.axes["energy_true"].edges[1:], "Energy hi")
        info += array_stats_str(self.axes["energy_true"].edges[:-1], "Energy lo")

        containment_radius = self.containment_radius(
            energy_true=energy_true, offset=offset, fraction=fraction
        )

        energy_true, offset, fraction = np.broadcast_arrays(
            energy_true, offset, fraction, subok=True
        )

        for idx in np.ndindex(containment_radius.shape):
            info += f"{100 * fraction[idx]:.2f} containment radius "
            info += f"at offset = {offset[idx]} "
            info += f"and energy_true = {energy_true[idx]:4.1f}: "
            info += f"{containment_radius[idx]:.3f}\n"

        return info

    def plot_containment_vs_energy(
            self, ax=None, fraction=[0.68, 0.95], offset=[0, 1] * u.deg,  **kwargs
    ):
        """Plot containment fraction as a function of energy.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`
            Axes to plot on.
        fraction : list of float or `~numpy.ndarray`
            Containment fraction between 0 and 1.
        offset : `~astropy.units.Quantity`
            Offset array
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.plot`

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
             Axes to plot on.

        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        energy_true = self.axes["energy_true"].center

        for theta in offset:
            for frac in fraction:
                plot_kwargs = kwargs.copy()
                radius = self.containment_radius(
                    energy_true=energy_true, offset=theta, fraction=frac
                )
                plot_kwargs.setdefault(
                    "label", f"{theta}, {100 * frac:.1f}%"
                )
                ax.plot(energy_true, radius, **plot_kwargs)

        ax.semilogx()
        ax.legend(loc="best")
        ax.set_xlabel(f"Energy ({energy_true.unit})")
        ax.set_ylabel(f"Containment radius ({radius.unit})")
        return ax

    def plot_containment(self, ax=None, fraction=0.68,  add_cbar=True, **kwargs):
        """Plot containment image with energy and theta axes.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`
            Axes to plot on.
        fraction : float
            Containment fraction between 0 and 1.
        add_cbar : bool
            Add a colorbar
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.pcolormesh`

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
             Axes to plot on.
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        energy = self.axes["energy_true"].center
        offset = self.axes["offset"].center

        # Set up and compute data
        containment = self.containment_radius(
            energy_true=energy[:, np.newaxis], offset=offset, fraction=fraction
        )

        # plotting defaults
        kwargs.setdefault("cmap", "GnBu")
        kwargs.setdefault("vmin", np.nanmin(containment.value))
        kwargs.setdefault("vmax", np.nanmax(containment.value))

        # Plotting
        x = energy.value
        y = offset.value
        caxes = ax.pcolormesh(x, y, containment.value.T, **kwargs)

        # Axes labels and ticks, colobar
        ax.semilogx()
        ax.set_ylabel(f"Offset ({offset.unit})")
        ax.set_xlabel(f"Energy ({energy.unit})")
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())

        if add_cbar:
            label = f"Containment radius R{100 * fraction:.0f} ({containment.unit})"
            ax.figure.colorbar(caxes, ax=ax, label=label)

        return ax

    def peek(self, figsize=(15, 5)):
        """Quick-look summary plots."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)

        self.plot_containment(fraction=0.68, ax=axes[0])
        self.plot_containment(fraction=0.95, ax=axes[1])
        self.plot_containment_vs_energy(ax=axes[2])
        plt.tight_layout()
