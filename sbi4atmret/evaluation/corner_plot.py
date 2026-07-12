"""
Corner plot evaluation with derived parameters.

Converts theta tensors to named dicts using posterior_names,
then selects parameters by name for plotting.

Includes a self-contained corner_mod function (no external dependencies
beyond lampe.plots and matplotlib).
"""

from dataclasses import dataclass
from typing import Optional, Any, List, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from lampe.plots import corner, LinearAlphaColormap
from petitRADTRANS import nat_cst as nc
import petitRADTRANS as prt

from sbi4atmret.utils.general import theta_to_dict
from sbi4atmret.utils.plotting_utils import legends


# =========================================================
# RESULT
# =========================================================

@dataclass(frozen=True)
class CornerResult:
    figure: Any
    theta_processed: list


# =========================================================
# DERIVED PARAMETER FUNCTIONS
# =========================================================

def ratio_14N_15N(theta_dict):
    """Compute 14N/15N isotope ratio from named dict."""
    N14 = 10 ** theta_dict["NH3_mol_scale"]
    N15 = 10 ** theta_dict["15NH3_mol_scale"]
    return (N14 * 18.02) / (N15 * 17.027)


def compute_log_gravity(theta_dict):
    """Compute log10(g) from R_pl and Mass."""
    gravity = nc.G * (theta_dict["mass"] * prt.nat_cst.m_jup) / (theta_dict["R_pl"] * prt.nat_cst.r_jup_mean) ** 2
    return torch.log10(gravity)


def compute_mass_from_logg(theta_dict):
    """Compute mass from R_pl and log g."""
    radius = theta_dict["R_pl"] * prt.nat_cst.r_jup_mean
    gravity = 10 ** theta_dict["$\\log g$"]
    mass = (gravity * radius ** 2) / nc.G
    return mass / prt.nat_cst.m_jup


# =========================================================
# CORNER_MOD (self-contained)
# =========================================================

def corner_mod(
    theta,
    weights=None,
    legend=None,
    color=None,
    figsize=(10, 10),
    domain=None,
    labels=None,
    labelsize=12,
    titlesize=14,
    fontsize=10,
    legend_fontsize=12,
    xtick_labelsize=10,
    ytick_labelsize=10,
    loc="center",
    bbox_to_anchor=(0.4, 0.9),
    labl=True,
    alpha=(0, 0.9),
):
    """
    Modified corner plot supporting multiple posteriors overlaid.

    Uses lampe.plots.corner internally with credible-region bands.

    Args:
        theta: list of (N, D) arrays, one per posterior
        weights: list of weight arrays (or None per entry)
        legend: list of legend strings
        color: list of color strings
        figsize: (w, h) figure size
        domain: (lower_tuple, upper_tuple) for axis limits
        labels: tuple of parameter labels
        labelsize, titlesize, fontsize, etc.: styling
        loc, bbox_to_anchor: legend placement
        labl: whether to show credible region legend
        alpha: (min, max) alpha for colormap

    Returns:
        matplotlib Figure
    """

    if legend is None:
        legend = [f"Post {i}" for i in range(len(theta))]
    if color is None:
        color = ["steelblue", "darkorange", "seagreen", "firebrick"][:len(theta)]
    if weights is None:
        weights = [None] * len(theta)

    params = {
        "axes.labelsize": labelsize,
        "axes.titlesize": titlesize,
        "font.size": fontsize,
        "legend.fontsize": legend_fontsize,
        "xtick.labelsize": xtick_labelsize,
        "ytick.labelsize": ytick_labelsize,
    }
    plt.rcParams.update(params)

    # Create a dummy figure for legend handles
    figure_dummy, axes_dummy = plt.subplots(
        figsize[0], figsize[0], squeeze=False, sharex="col",
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    )
    plt.close(figure_dummy)

    # Build corner plot by overlaying each posterior
    fig = None
    for i, th in enumerate(theta):
        corner_kwargs = dict(
            smooth=2,
            domain=domain,
            labels=labels,
            figsize=figsize,
            creds=[0.997, 0.955, 0.683],
            alpha=alpha,
            color=color[i],
            figure=fig,
        )
        if weights[i] is not None:
            corner_kwargs["weights"] = weights[i]
        fig = corner(th, **corner_kwargs)

    # Fix axis labels
    if labels is not None:
        for index, ax in enumerate(fig.get_axes()):
            ax.tick_params(axis="both")

    plt.subplots_adjust(bottom=0.15)

    # Clear any auto-generated legends
    fig.legends.clear()
    for ax in fig.get_axes():
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    # Build legend handles on dummy axes
    for i in range(len(legend)):
        axes_dummy[0, -1].plot([], [], color=color[i], label=legend[i])

    if labl:
        handles, texts = legends(axes=axes_dummy, alpha=alpha, color=color[0])
        fig.legend(
            handles, texts, loc=loc, bbox_to_anchor=bbox_to_anchor,
            bbox_transform=fig.transFigure, frameon=False,
        )
    else:
        handles, texts = axes_dummy[0, -1].get_legend_handles_labels()
        fig.legend(
            handles, texts, loc=loc, bbox_to_anchor=bbox_to_anchor,
            bbox_transform=fig.transFigure, frameon=False,
        )

    return fig


# =========================================================
# CORNER EVALUATOR
# =========================================================

class CornerEvaluator:
    """
    Corner plot evaluator.

    Converts theta → named dict via posterior_names,
    selects requested parameters by name,
    appends derived quantities,
    and plots.
    """

    def plot_corner(
        self,
        theta_list: List[torch.Tensor],
        posterior_names: List[str],
        param_names_to_plot: List[str] = None,
        derived_params: List[Dict] = None,
        legends_list: List[str] = None,
        colors: List[str] = None,
        theta_star: Optional[List[torch.Tensor]] = None,
        n_samples: int = 20469,
        **kwargs,
    ):
        """
        Create a corner plot by selecting parameters by name.

        Args:
            theta_list: list of (N, D) posterior sample tensors
            posterior_names: list of D parameter names matching theta columns
            param_names_to_plot: which parameters to plot (by name).
                                 If None, plots all.
            derived_params: list of dicts:
                - "name": label
                - "fn": callable(theta_dict) -> (N,) tensor
                - "lower": float
                - "upper": float
            legends_list: legend strings per posterior
            colors: colors per posterior
            theta_star: optional ground truth tensors to mark
            n_samples: max samples per posterior
            **kwargs: passed to corner_mod (labelsize, alpha, etc.)

        Returns:
            matplotlib Figure
        """
        from lampe.plots import mark_point

        if derived_params is None:
            derived_params = []
        if legends_list is None:
            legends_list = [f"Posterior {i}" for i in range(len(theta_list))]
        if colors is None:
            colors = ["steelblue", "darkorange", "seagreen", "firebrick"][:len(theta_list)]

        # If no subset specified, use all
        if param_names_to_plot is None:
            param_names_to_plot = list(posterior_names)

        # Get bounds from config
        prior_params = {p.name: p for p in self.config.prior_config.parameters}

        # Process each posterior
        processed_thetas = []
        final_labels = []
        final_lower = []
        final_upper = []
        built_labels = False

        for theta in theta_list:
            td = theta_to_dict(theta[:n_samples], posterior_names)

            selected_values = []
            for name in param_names_to_plot:
                if name in td:
                    selected_values.append(td[name])
                    if not built_labels:
                        final_labels.append(name)
                        final_lower.append(prior_params[name].lower)
                        final_upper.append(prior_params[name].upper)

            for dp in derived_params:
                derived_val = dp["fn"](td)
                selected_values.append(derived_val[:n_samples])
                if not built_labels:
                    final_labels.append(dp["name"])
                    final_lower.append(dp["lower"])
                    final_upper.append(dp["upper"])

            built_labels = True
            processed_thetas.append(torch.stack(selected_values, dim=-1))

        # Plot
        fig = corner_mod(
            theta=[t.numpy() for t in processed_thetas],
            legend=legends_list,
            color=colors,
            domain=(tuple(final_lower), tuple(final_upper)),
            labels=tuple(final_labels),
            **kwargs,
        )

        # Mark ground truth
        if theta_star is not None:
            for ts in theta_star:
                td_star = theta_to_dict(ts.unsqueeze(0), posterior_names)
                star_vals = []
                for name in param_names_to_plot:
                    if name in td_star:
                        star_vals.append(td_star[name].item())
                for dp in derived_params:
                    star_vals.append(dp["fn"](td_star).item())
                mark_point(fig, np.array(star_vals), color="black")

        return fig
