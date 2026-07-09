from functools import partial
import numpy as np
from numpy import ndarray as Array
from scipy.interpolate import interp1d, CubicSpline
import petitRADTRANS.retrieval.parameter as prm
from petitRADTRANS import poor_mans_nonequ_chem as pm
from typing import * 
import copy as cp


def pt_profile(theta: Union[Array, Dict], pressures: Array) -> Array:
    ## the temperature of the Nth layer is - fraction1* fraction2* .....fraction N * Tbottom 

    if isinstance(theta, np.ndarray):
        r"""Returns the pressure-temperature profile."""
        temp_nodes = theta[3:12]
        node_locations = np.logspace(-6.01, 3.01, len(temp_nodes)+1)
        node_loc_log = np.log10(node_locations)
        t_bottom= theta[2]
        node_temps = np.zeros_like(node_locations)
        node_temps[-1] = t_bottom
        for i_temp in range(len(temp_nodes)):
            node_temps[-2 - i_temp] = temp_nodes[i_temp] * node_temps[-1 - i_temp]

    elif isinstance(theta, dict):
        node_locations = np.logspace(-6.01, 3.01, theta['N_nodes'].value)
        node_loc_log = np.log10(node_locations)
        t_bottom = theta['T_bottom'].value
        node_temps = np.zeros_like(node_locations)
        node_temps[-1] = t_bottom
        for i_temp in range(theta['N_nodes'].value - 1):
            node_temps[-2 - i_temp] = theta['temp_node_'+str(i_temp+1)].value * node_temps[-1 - i_temp]

    else: 
        raise ValueError('theta must be a NumPy array or a dictionary')
    
    return temp_model_nodes(np.log10(pressures), node_temps, node_loc_log, 'quadratic') 


def temp_model_nodes(log_pressures,
               node_temps,
               node_locations,
               kind):


    temps = interp1d(node_locations,
                     node_temps,
                     kind)

    retval = temps(log_pressures)

    # To prevent overshoot or undershoot wiggles of higher order interpolation
    '''
    search_sort = np.searchsorted(node_locations, log_pressures)
    index = retval < node_temps[search_sort - 1]
    retval[index] = node_temps[search_sort - 1][index]

    index = retval > node_temps[search_sort]
    retval[index] = node_temps[search_sort][index]
    '''

    # This here is then not needed anymore...
    index = retval < 10.
    retval[index] = 10.

    return retval


def PT_ret_model(theta: Union[Array, Dict], pressures: Array):
    """
    Self-luminous retrieval P-T model.

    Args:
        theta : Array or Dictionary
            input parameters
        pressures : np.ndarray
            input pressure profile in bar
    Returns:
        Tret : np.ndarray
            The temperature as a function of atmospheric pressure.
    """

    if isinstance(theta, np.ndarray):
        CO, FeH, *_, T_int, T3, T2, T1, alpha, log_delta = theta
        T3 = ((3 / 4 * T_int ** 4 * (0.1 + 2 / 3)) ** (1 / 4)) * (1 - T3)
        T2 = T3 * (1 - T2)
        T1 = T2 * (1 - T1)
        delta = (1e6 * 10 ** (-3 + 5 * log_delta)) ** (-alpha)
        conv = True

    elif isinstance(theta, dict):
        # Make the P-T profile
        alpha = theta['alpha'].value
        T_int = theta['T_int'].value
        CO = theta['CO'].value
        FeH = theta['FeH'].value
        conv = theta['conv']
        T3 = ((3./4.*T_int**4.*(0.1+2./3.))**0.25)*(1.0-theta['T3'].value)
        T2 = T3*(1.0-theta['T2'].value)
        T1 = T2*(1.0-theta['T1'].value)
        delta = ((10.0**(-3.0+5.0*theta['log_delta'].value))*1e6)**(-alpha)

    else: 
        raise ValueError('theta must be a NumPy array or a dictionary')
    
    # Go from bar to cgs
    press_cgs = pressures * 1e6

    # Calculate the optical depth
    tau = delta * press_cgs ** alpha

    # This is the eddington temperature
    tedd = (3. / 4. * T_int ** 4. * (2. / 3. + tau)) ** 0.25

    ab = pm.interpol_abundances(CO * np.ones_like(tedd),
                                FeH * np.ones_like(tedd),
                                tedd,
                                pressures)

    nabla_ad = ab['nabla_ad']

    # Enforce convective adiabat
    if conv:
        # Calculate the current, radiative temperature gradient
        nab_rad = np.diff(np.log(tedd)) / np.diff(np.log(press_cgs))
        # Extend to array of same length as pressure structure
        nabla_rad = np.ones_like(tedd)
        nabla_rad[0] = nab_rad[0]
        nabla_rad[-1] = nab_rad[-1]
        nabla_rad[1:-1] = (nab_rad[1:] + nab_rad[:-1]) / 2.

        # Where is the atmosphere convectively unstable?
        conv_index = nabla_rad > nabla_ad

        # TODO: Check remains convective and convergence
        for i in range(10):
            if i == 0:
                t_take = cp.copy(tedd)
            else:
                t_take = cp.copy(tfinal)  # TODO possible reference before assignment

            ab = pm.interpol_abundances(CO * np.ones_like(t_take),
                                        FeH * np.ones_like(t_take),
                                        t_take,
                                        pressures)

            nabla_ad = ab['nabla_ad']

            # Calculate the average nabla_ad between the layers
            nabla_ad_mean = nabla_ad
            nabla_ad_mean[1:] = (nabla_ad[1:] + nabla_ad[:-1]) / 2.
            # What are the increments in temperature due to convection
            tnew = nabla_ad_mean[conv_index] * np.mean(np.diff(np.log(press_cgs)))
            # What is the last radiative temperature?
            tstart = np.log(t_take[~conv_index][-1])
            # Integrate and translate to temperature from log(temperature)
            tnew = np.exp(np.cumsum(tnew) + tstart)

            # Add upper radiative and
            # lower conective part into one single array
            tfinal = cp.copy(t_take)
            tfinal[conv_index] = tnew

            if np.max(np.abs(t_take - tfinal) / t_take) < 0.01:
                break

    else:
        tfinal = tedd

    # Add the three temperature-point P-T description above tau = 0.1
    def press_tau(tau):
        # Returns the pressure at a given tau, in cgs
        return (tau / delta) ** (1. / alpha)

    # Where is the uppermost pressure of the Eddington radiative structure?
    p_bot_spline = press_tau(0.1)

    for i_intp in range(2):

        if i_intp == 0:

            # Create the pressure coordinates for the spline support nodes at low pressure
            support_points_low = np.logspace(np.log10(press_cgs[0]),
                                            np.log10(p_bot_spline),
                                            4)

            # Create the pressure coordinates for the spline support nodes at high pressure,
            # the corresponding temperatures for these nodes will be taken from the
            # radiative+convective solution
            support_points_high = 10 ** np.arange(np.log10(p_bot_spline),
                                                np.log10(press_cgs[-1]),
                                                np.diff(np.log10(support_points_low))[0])

            # Combine into one support node array, don't add the p_bot_spline point twice.
            support_points = np.zeros(len(support_points_low) + len(support_points_high) - 1)
            support_points[:4] = support_points_low
            support_points[4:] = support_points_high[1:]

        else:

            # Create the pressure coordinates for the spline support nodes at low pressure
            support_points_low = np.logspace(np.log10(press_cgs[0]),
                                            np.log10(p_bot_spline),
                                            7)

            # Create the pressure coordinates for the spline support nodes at high pressure,
            # the corresponding temperatures for these nodes will be taken from the
            # radiative+convective solution
            support_points_high = np.logspace(np.log10(p_bot_spline), np.log10(press_cgs[-1]), 7)

            # Combine into one support node array, don't add the p_bot_spline point twice.
            support_points = np.zeros(len(support_points_low) + len(support_points_high) - 1)
            support_points[:7] = support_points_low
            support_points[7:] = support_points_high[1:]

        # Define the temperature values at the node points.
        t_support = np.zeros_like(support_points)

        if i_intp == 0:
            tfintp = interp1d(press_cgs, tfinal, kind='cubic')
            # The temperature at p_bot_spline (from the radiative-convectice solution)
            t_support[int(len(support_points_low)) - 1] = tfintp(p_bot_spline)
            # The temperature at pressures below p_bot_spline (free parameters)
            t_support[:(int(len(support_points_low)) - 1)] = T3
            # t_support[:3] = tfintp(support_points_low)
            # The temperature at pressures above p_bot_spline
            # (from the radiative-convectice solution)
            t_support[int(len(support_points_low)):] = \
                tfintp(support_points[(int(len(support_points_low))):])

        else:
            tfintp1 = interp1d(press_cgs, tret, kind='cubic')  # TODO possible reference before assignment
            t_support[:(int(len(support_points_low)) - 1)] = \
                tfintp1(support_points[:(int(len(support_points_low)) - 1)])

            tfintp = interp1d(press_cgs, tfinal)
            # The temperature at p_bot_spline (from the radiative-convectice solution)
            t_support[int(len(support_points_low)) - 1] = tfintp(p_bot_spline)
            # print('diff', t_connect_calc - tfintp(p_bot_spline))
            t_support[int(len(support_points_low)):] = \
                tfintp(support_points[(int(len(support_points_low))):])

        # Make the temperature spline interpolation to be returned to the user
        cs = CubicSpline(np.log10(support_points), t_support)
        tret = cs(np.log10(press_cgs))

    tret[tret < 0.0] = 10.0
    # Return the temperature, the pressure at tau = 1,
    # and the temperature at the connection point.
    # The last two are needed for the priors on the P-T profile.
    return tret  # , press_tau(1.)/1e6, tfintp(p_bot_spline)
