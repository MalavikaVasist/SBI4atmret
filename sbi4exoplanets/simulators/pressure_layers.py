import numpy as np
import sys


def initialize_pressure_layers(press, parameters): 
        if parameters['AMR']:
            try:
                pglobal_check(press, ## here the press is in bar 
                        parameters['pressure_simple'].value,
                        parameters['pressure_scaling'].value)
            except KeyError():
                print("You must include the pressure_simple and pressure_scaling parameters when using AMR!")
                sys.exit(1)

            p_use = PGLOBAL ## len 1000 in bar
        else:
            p_use = press  ## len 127 in bar
        return p_use


def pglobal_check(press,shape,scaling):
    """
    Check to ensure that the global pressure array has the correct length.
    Updates PGLOBAL.

    Args:
        press : numpy.ndarray
            Pressure array from a pRT_object. Used to set the min and max values of PGLOBAL
        shape : int
            the shape of the pressure array if no AMR is used
        scaling :
            The factor by which the pressure array resolution should be scaled.
    """

    global PGLOBAL
    if PGLOBAL.shape[0] != int(scaling*shape):
        PGLOBAL = np.logspace(np.log10(press[0]),
                              np.log10(press[-1]),
                              int(scaling*shape)) #PGLOBAL = np.logspace(-6,3,1000) 





        


