from .context import DomainContext


def build_domain_context(simulators, observation, config):
    """
    Construct the shared scientific/domain context.

    This function centralizes:
    - simulator setup
    - observation metadata
    - wavelength bookkeeping
    - parameter indexing
    - preprocessing pipeline
    - noise model construction

    Returns
    -------
    DomainContext
    """
     
    # -----------------------------------
    # parameter indexing
    # -----------------------------------
    '''
    Maps parameter name → column index in theta
    '''
    
    param_index = {
        simname: {
            name: i for i, name in enumerate(sim.names)
        }
        for simname, sim in simulators.items()
    }

    # -----------------------------------
    # simulator metadata
    # -----------------------------------
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

    # -----------------------------------
    # observation metadata
    # -----------------------------------

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

    # -----------------------------------
    # misc metadata
    # -----------------------------------

    scale = getattr(
        config.dataset_config,
        "scale",
        1.0,
    )

    unsort_index = getattr(
        observation,
        "unsort_index",
        None,
    )

    # -----------------------------------
    # temporary partial domain
    # needed because pipe/noise depend on it
    # -----------------------------------

    partial_domain = DomainContext(
        simulators=simulators,
        observation=observation,

        pipe=None,
        noise=None,

        param_index=param_index,

        sim_wlens=sim_wlens,
        obs_wlens=obs_wlens,

        obs_noise=obs_noise,

        scale=scale,

        unsort_index=unsort_index,
    )

    # -----------------------------------
    # build domain-dependent components
    # -----------------------------------

    pipe = config.build_pipe(
        domain=partial_domain
    )

    noise = config.build_noise(
        domain=partial_domain
    )

    # -----------------------------------
    # final immutable domain
    # -----------------------------------

    return DomainContext(
        simulators=simulators,
        observation=observation,

        pipe=pipe,
        noise=noise,

        param_index=param_index,

        sim_wlens=sim_wlens,
        obs_wlens=obs_wlens,

        obs_noise=obs_noise,

        scale=scale,

        unsort_index=unsort_index,
    )

    
