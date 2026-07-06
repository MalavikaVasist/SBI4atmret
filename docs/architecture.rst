Package Architecture
====================

Overview
--------

.. code-block:: text

   sbi4atmret/
   ├── config/          # Pydantic config models, YAML parsing
   ├── datasets/        # Dataset loading, pipes, theta mapping
   ├── domain/          # Domain context (simulators, observation, pipe, noise)
   ├── estimator/       # Neural network: embedding + normalizing flow
   ├── evaluation/      # All evaluation methods (coverage, consistency, PT, corner, ...)
   ├── likelihoods/     # Noise models (Gaussian heteroscedastic)
   ├── models/          # Model composition (BaseModel, meta-learner)
   ├── observations/    # Observation loading and metadata
   ├── runtime/         # Batch processing, runtime context setup
   ├── simulators/      # petitRADTRANS-based atmospheric simulators
   ├── training/        # Training loop, setup, losses
   └── utils/           # Checkpointing, plotting utilities, general helpers

Module Relationships
--------------------

.. code-block:: text

   BaseConfig (YAML)
       │
       ├── build_simulators() → {sim_name: Simulator}
       ├── build_prior() → BoxUniform
       ├── build_pipe(domain) → BasePipe subclass
       ├── build_noise(domain) → GaussianNoise
       ├── build_embedding() → SoftclipResMLP
       ├── build_flow() → NPEFlow
       └── build_loss(estimator) → BNPELoss
                │
                ▼
   DomainContext (frozen dataclass)
       ├── simulator_dict
       ├── observation
       ├── pipe
       ├── noise
       ├── sim_param_index
       ├── sim_wlens / obs_wlens / obs_noise
       └── scale, unsort_index

Data Flow (Training)
--------------------

.. code-block:: text

   H5Dataset (per simulator)
       │
       ▼
   DataLoader → batches: [(theta, x), ...]
       │
       ▼
   Dataset.reconstruct_batch(keys, batches) → batch_dict: {sim_name: (theta, x)}
       │
       ▼
   BatchProcessor.prepare_batch(batch_dict)
       ├── pipe.forward(batch_dict, mode="train")
       │     ├── modify_spec()  — trim, mask, rebin
       │     └── modify_theta() — transform b-factors
       ├── noise(processed) — add heteroscedastic noise
       └── pipe.build_input(processed) → merge to (theta_merged, x_merged)
           │
           ▼
       to_device() → (theta, x) on GPU
           │
           ▼
       loss_fn(theta, x) → backprop

Data Flow (Evaluation)
----------------------

.. code-block:: text

   BaseEvaluator.__init__()
       ├── load checkpoint → net.to(device)
       ├── build posterior from x_obs
       └── sample theta from posterior (or load from CSV)
           │
           ▼
       run_coverage()   — SBC ranks on test set
       run_consistency() — simulate from posterior, compute residuals
       run_PT()         — PT profiles + contribution from MAP sample
       run_corner()     — corner plots with derived parameters
       run_bolometric() — T_eff and luminosity

Estimator Architecture
----------------------

.. code-block:: text

   EstimatorBase(nn.Module)
       ├── embedding: SoftclipResMLP
       │     ├── SoftClip(bound=100)
       │     └── ResMLP(input_dim → output_dim)
       │           per instrument (miri: 1298→64, gemini: 434→16)
       │
       └── flow: NPEFlow
             └── lampe.inference.NPE(n_params, emb_dim, transforms=5)

   Forward:
       x_obs → embedding(x) → x_emb → flow(theta, x_emb) → log_prob
       x_obs → embedding(x) → x_emb → flow.flow(x_emb) → posterior distribution
