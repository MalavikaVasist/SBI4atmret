from .context import DomainContext


def build_domain_context(simulators, observation):

    '''
    Maps parameter name → column index in theta
    '''
    param_index = {
        simname: {
            name: i for i, name in enumerate(sim.names)
        }
        for simname, sim in simulators.items()
    }

    '''
    returns:
    sim_wlens = {
            "cloudfree_miri" : wlen, 
            "cloudfree_hst": wlen, 
            "cloudfre_gemini" : wlen, 

            "..": ....

                }
    '''

    sim_wlens = {
        name: sim.wavelength
        for name, sim in simulators.items()
    }

    '''
    returns:
    obs_wlens = {
            "miri" : wlen, 
            "hst": wlen, 
            "gemini" : wlen, 
                }
        '''

    obs_data = observation.instrument_metadata

    obs_wlens = {
        inst: d["wlen"]
        for inst, d in obs_data.items()
    }

    '''
    returns:
    obs_noise = {
            "miri" : sigmaM, 
            "hst": sigmaH, 
            "gemini" : sigmaG, 
                }
    '''

    obs_noise = {
        inst: d["sigma"]
        for inst, d in obs_data.items()
    }


    return DomainContext(
        simulators=simulators,
        observation=observation,
        param_index=param_index,
        sim_wlens=sim_wlens,
        obs_wlens=obs_wlens,
        obs_noise=obs_noise,
        scale=observation.scale,
        unsort_index = observation.unsort_index, 
        
    )