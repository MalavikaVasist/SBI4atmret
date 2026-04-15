import sys
import os
import time
import copy as cp
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from petitRADTRANS import nat_cst as nc
from typing import Tuple
from petitRADTRANS.retrieval.util import calc_MMW, surf_to_meas
from petitRADTRANS.retrieval import cloud_cond as fc
from petitRADTRANS import poor_mans_nonequ_chem as pm

"""
Models Module

This module contains a set of functions that generate the spectra used
in the petitRADTRANS retrieval. This includes setting up the
pressure-temperature structure, the chemistry, and the radiative
transfer to compute the emission or transmission spectrum.

All models must take the same set of inputs:

    pRT_object : petitRADTRANS.RadTrans
        This is the pRT object that is used to compute the spectrum
        It must be fully initialized prior to be used in the model function
    parameters : dict
        A dictionary of Parameter objects. The naming of the parameters
        must be consistent between the Priors and the model function you
        are using.
    PT_plot_mode : bool
        If this argument is True, the model function should return the pressure
        and temperature arrays before computing the flux.
    AMR : bool
        If this parameter is True, your model should allow for reshaping of the
        pressure and temperature arrays based on the position of the clouds or
        the location of the photosphere, increasing the resolution where required.
        For example, using the fixed_length_amr function defined below.
"""


def emission_model_diseq(pRT_object,
                         parameters,
                         p_use,
                         temperatures, 
                         ):



    abundances, MMW, small_index, Pbases = get_abundances(p_use,
                                                  temperatures,
                                                  pRT_object.line_species,
                                                  pRT_object.cloud_species,
                                                  parameters,
                                                  AMR =parameters['AMR'])
    
    
    # Only include the high resolution pressure array near the cloud base.
    if abundances is None:
        return None, None
    
    if parameters['AMR']:
        temperatures = temperatures[small_index]
        pressures = p_use[small_index] ## AMR meshgrid of p_use -> 1000[meshgrid] in bar
        MMW = MMW[small_index]
        pRT_object.press = pressures * 1e6 ##in cgs- converted from bar
    else:
        pressures = p_use ##127 in bar

    if parameters['PT_plot_mode']:
        return pressures, temperatures

    ##
    gravity = -np.inf
    if 'log_g' in parameters.keys() and 'mass' in parameters.keys():
        gravity = 10**parameters['log_g'].value
        R_pl = np.sqrt(nc.G*parameters['mass'].value/gravity)
    elif 'log_g' in parameters.keys():
        gravity= 10**parameters['log_g'].value
        R_pl = parameters['R_pl'].value
    elif 'mass' in parameters.keys():
        R_pl = parameters['R_pl'].value
        gravity = nc.G * parameters['mass'].value/R_pl**2
    else:
        print("Pick two of log_g, R_pl and mass priors!")
        sys.exit(5)

    # Hansen or log normal clouds
    if len(pRT_object.cloud_species):
        sigma_lnorm, fseds, kzz, b_hans, radii, distribution = fc.setup_clouds(pressures, parameters, pRT_object.cloud_species)
    else: 
        sigma_lnorm, fseds, kzz, b_hans, radii, distribution = None

    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value

    # Calculate the spectrum
    pRT_object.calc_flux(temperatures,
                        abundances,
                        gravity,
                        MMW,
                        contribution =  contribution,
                        fsed = fseds,
                        Kzz = kzz,
                        sigma_lnorm = sigma_lnorm,
                        b_hans = b_hans,
                        radius = radii,
                        dist = distribution, 
                        )

    # Getting the model into correct units (Jy)
    wlen_model = nc.c/pRT_object.freq/1e-4
    f_nu = pRT_object.flux*1e23

    spectrum_model = surf_to_meas(f_nu,
                                  R_pl,
                                  parameters['D_pl'].value)
    
    
    if contribution:
        return wlen_model, spectrum_model, np.abs(pRT_object.contr_em)
    else: 
        return wlen_model, spectrum_model



def fixed_length_amr(p_clouds, pressures, scaling = 10, width = 3):
    r"""This function takes in the cloud base pressures for each cloud,
    and returns an array of pressures with a high resolution mesh
    in the region where the clouds are located.

    Author:  Francois Rozet.

    The output length is always
        len(pressures[::scaling]) + len(p_clouds) * width * (scaling - 1)

    Args:
        P_clouds : numpy.ndarray
            The cloud base pressures in bar
        press : np.ndarray
            The high resolution pressure array.
        scaling : int
            The factor by which the low resolution pressure array is scaled
        width : int
            The number of low resolution bins to be replaced for each cloud layer.
    """

    length = len(pressures)
    cloud_indices = np.searchsorted(pressures, np.asarray(p_clouds))

    # High resolution intervals
    def bounds(center: int, width: int) -> Tuple[int, int]:
        upper = min(center + width // 2, length)
        lower = max(upper - width, 0)
        return lower, lower + width

    intervals = [bounds(idx, scaling * width) for idx in cloud_indices]

    # Merge intervals
    while True:
        intervals, stack = sorted(intervals), []

        for interval in intervals:
            if stack and stack[-1][1] >= interval[0]:
                last = stack.pop()
                interval = bounds(
                    (last[0] + max(last[1], interval[1]) + 1) // 2,
                    last[1] - last[0] + interval[1] - interval[0],
                )

            stack.append(interval)

        if len(intervals) == len(stack):
            break
        intervals = stack

    # Intervals to indices
    indices = [np.arange(0, length, scaling)]

    for interval in intervals:
        indices.append(np.arange(*interval))

    indices = np.unique(np.concatenate(indices))

    return pressures[indices], indices


def get_abundances(pressures, temperatures, line_species, cloud_species, parameters):
    """
    This function takes in the C/O ratio, metallicity, and quench pressures and uses them
    to compute the gas phase and equilibrium condensate abundances from an interpolated table.
    This function assumes a hydrogen-helium dominated atmosphere, and enforces <10% trace gas
    abundance by mass.

    Args:
        pressures : numpy.ndarray
            A log spaced pressure array. If AMR is on it should be the full high resolution grid.
        temperatures : numpy.ndarray
            A temperature array with the same shape as pressures
        line_species : List(str)
            A list of gas species that will contribute to the line-by-line opacity of the pRT atmosphere.
        cloud_species : List(str)
            A list of condensate species that will contribute to the cloud opacity of the pRT atmosphere.
        parameters : dict
            A dictionary of model parameters, in particular it must contain the names C/O, Fe/H and
            log_pquench. Additionally the cloud parameters log_X_cb_Fe(c) and MgSiO3(c) must be present.
        AMR : bool
            Turn the adaptive mesh grid on or off. See fixed_length_amr for implementation.

    Returns:
        abundances : dict
            Mass fraction abundances of all atmospheric species
        MMW : numpy.ndarray
            Array of the mean molecular weights in each pressure bin
        small_index : numpy.ndarray
            The indices of the high resolution grid to use to define the adaptive grid.
        PBases : dict
            A dictionary of the cloud base pressures, either computed from equilibrium
            condensation or set by the user.
    """

    abundances_interp = {}

    # Equilibrium chemistry - means no advection/mixing/photochemistry
    if "C/O" in parameters.keys():
        # Make the abundance profile
        pquench_C = None
        if 'log_pquench' in parameters.keys():
            pquench_C = 10**parameters['log_pquench'].value
        abundances_interp = pm.interpol_abundances(parameters['C/O'].value * np.ones_like(pressures), \
                                                parameters['Fe/H'].value * np.ones_like(pressures), \
                                                temperatures, \
                                                pressures,
                                                Pquench_carbon = pquench_C)
        MMW = abundances_interp['MMW']
    
    # Free chemistry abundances
    else: 
        msum = 0.0
        # print(parameters.keys())
        for species in line_species:        
            if species.split('_')[0]+'_mol_scale' in parameters.keys():
                abund = 10**parameters[species.split('_')[0]+'_mol_scale'].value
                abundances_interp[species.split('_')[0]] = abund * np.ones_like(pressures)
                msum += abund

            elif ('Na_' in species):
                abundances_interp[species.split('_')[0]] = 0.9 * \
                                        1e1 ** parameters['alkali_mol_scale'].value * np.ones_like(pressures)
                msum += 0.9 * 1e1 ** parameters['alkali_mol_scale'].value
            elif ('K_' in species):
                abundances_interp[species.split('_')[0]] = 0.1 * \
                                        1e1 ** parameters['alkali_mol_scale'].value * np.ones_like(pressures)
                msum += 0.1 * 1e1 ** parameters['alkali_mol_scale'].value

            elif species.split("_R_")[0] in parameters.keys():
                # Cannot mix free and equilibrium chemistry. Maybe something to add?
                abund = 10**parameters[species.split("_R_")[0]].value
                abundances_interp[species.split('_')[0]] = abund * np.ones_like(pressures)
                msum += abund

        # Whatever's left is H2 and
        abundances_interp['H2'] = 0.766 * (1.0-msum) * np.ones_like(pressures)
        abundances_interp['He'] = 0.234 * (1.0-msum) * np.ones_like(pressures)
        # Imposing strict limit on msum to ensure H2 dominated composition
        if msum > 1.0:
                print(f"Abundance sum > 1.0, msum={msum}")
                return None,None,None,None
        
        MMW = calc_MMW(abundances_interp)

    # Prior check all input params

    # Here you see how much clouds there are
    clouds = {}
    Pbases = {}

    for cloud in cloud_species:
        cname = cloud.split("_")[0]
        if "eq_scaling_"+cname in parameters.keys():
            # equilibrium cloud abundance
            Xcloud= fc.return_cloud_mass_fraction(cloud,parameters['Fe/H'].value, parameters['C/O'].value)
            # Scaled by a constant factor
            clouds[cname] = 10**parameters['eq_scaling_'+cname].value*Xcloud
        else:
            # Free cloud abundance
            clouds[cname] = 10**parameters['log_X_cb_'+cloud.split("_")[0]].value

        # Get the cloud locations
        # Here you see where to put those clouds 

        # Free cloud bases
        if 'log_Pbase_'+cname in parameters.keys():
            Pbases[cname] = 10**parameters['log_Pbase_'+cname].value
        elif 'Pbase_'+cname in parameters.keys():
            Pbases[cname] = parameters['Pbase_'+cname].value
        # Equilibrium locations
        elif 'Fe/H' in parameters.keys():
            Pbases[cname] = fc.simple_cdf(cname, pressures, temperatures,
                                            parameters['Fe/H'].value, parameters['C/O'].value, np.mean(MMW))
        else:
            Pbases[cname] = fc.simple_cdf_free(cname,
                                               pressures,
                                               temperatures,
                                               10**parameters['log_X_cb_'+cname].value,
                                               MMW[0])
            
    # print('Pbase ', Pbases)
    # Find high resolution pressure grid and indices
    if parameters.get('AMR') and len(Pbases) > 0:
        _, small_index = fixed_length_amr(np.array(list(Pbases.values())),
                                                  pressures,
                                                  parameters['pressure_scaling'].value,
                                                  parameters['pressure_width'].value) ##keep highres mesh only around clouds
    else :
        small_index = np.linspace(0,pressures.shape[0]-1,pressures.shape[0], dtype = int) ## keep all 1000 layers

    ## Here you see how to spread those clouds in the atmosphere
    fseds = {}
    abundances = {}
    for cloud in cp.copy(cloud_species):
        cname = cloud.split('_')[0]
        # Set up fseds per-cloud
        if 'fsed_'+cname in parameters.keys():
            fseds[cname] = parameters['fsed_'+cname].value
        else:
            fseds[cname] = parameters['fsed'].value
        abundances[cname] = np.zeros_like(temperatures)
        # print(pressures[pressures < Pbases[cname]], pressures[pressures <= Pbases[cname]])
        # abundances[cname][pressures < Pbases[cname]] = \
        abundances[cname][pressures <= Pbases[cname]] = \
                        clouds[cname] *\
                        (pressures[pressures <= Pbases[cname]]/\
                        Pbases[cname])**fseds[cname]
        abundances[cname] = abundances[cname][small_index]

    for species in line_species:
        if 'FeH' in species and 'Fe(c)' in Pbases:
            # Magic factor for FeH opacity - off by factor of 2
            abunds_change_rainout = cp.copy(abundances_interp[species.split('_')[0]]/2.)
            index_ro = pressures < Pbases['Fe(c)'] 
            abunds_change_rainout[index_ro] = 0.
            abundances[species] = abunds_change_rainout[small_index]
        else:    
            abundances[species] = abundances_interp[species.split('_')[0]][small_index] ##abundances reduced to the AMR grid for line species too if AMR = True
    abundances['H2'] = abundances_interp['H2'][small_index]
    abundances['He'] = abundances_interp['He'][small_index]

    return abundances, MMW, small_index, Pbases


