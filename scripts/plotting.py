from ..sbi4exoplanets.Plots.Plots import plots, ratio, computing_gravity, computing_mass
from added_scripts.corner_modified import cornerWratio_notfull


def plot_results(runpath, estimator, observation, testsets, pipe, simulator, config: Config):
    plot = plots(runpath, 0, estimator, observation)  # epoch is 0 for final

    cov_fig = plot.coverage(testsets, pipe, simulator)

    LABELS = [p[0] for p in config["PARAMETERS"]]
    LOWER = [p[1] for p in config["PARAMETERS"]]
    UPPER = [p[2] for p in config["PARAMETERS"]]

    appending_params_dict = {
        r'$^{14}N/^{15}N$': {"limits": [0, 1000], "method": ratio},
        r'$log g$': {"limits": [2, 6], "method": computing_gravity},
        r'$Mass$': {"limits": [1, 50], "method": computing_mass}
    }
    corner_fig = cornerWratio_notfull(
        LOWER, UPPER, LABELS, theta=None,
        columns=[0, 1, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        appending_params_dict=appending_params_dict,
        legends=['NPE'], colors=['steelblue'], savepath=None,
        labelsize=18, titlesize=20, fontsize=16, legend_fontsize=20,
        xtick_labelsize=28, ytick_labelsize=28,
        theta_star=None, loc='center', bbox_to_anchor=(0.4, 0.9),
        labl=True, alpha=[0, 0.9]
    )

    fig_pt = plot.ptprofile()
    res_fig_miri = plot.consistencyplot_MIRI()
    res_fig_gemini = plot.consistencyplot_Gemini()
    res_fig_hst = plot.consistencyplot_HST()
    cornerWratio_fig = plot.cornerWratio()

    return {
        'coverage': cov_fig,
        'corner': corner_fig,
        'pt_profile': fig_pt,
        'res_fig_miri': res_fig_miri,
        'res_fig_gemini': res_fig_gemini,
        'res_fig_hst': res_fig_hst,
        'cornerWratio_fig': cornerWratio_fig
    }