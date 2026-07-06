Quick Start
===========

SBI4atmret follows a three-stage workflow: **Generate → Train → Evaluate**.

1. Generate Dataset
-------------------

.. code-block:: bash

   python scripts/generate_dataset.py \
       --config experiments/config_MiriGeminiHST_cloudfree.yaml \
       --output-dir /path/to/simulations/ \
       --n-batches 300 \
       --batch-size 4096

This samples parameters from the prior, runs each simulator, filters
invalid spectra, and stores as HDF5 files split into train/valid/test.

2. Train
--------

.. code-block:: bash

   python scripts/training_evaluation.py \
       --config experiments/config_MiriGeminiHST_cloudfree.yaml \
       --action train

The trainer:

- Loads HDF5 datasets per instrument
- Processes batches through the pipe (spectral transforms + noise)
- Trains the NPE flow with gradient descent
- Logs metrics to Weights & Biases
- Saves checkpoints periodically

3. Evaluate
-----------

.. code-block:: bash

   python scripts/training_evaluation.py \
       --config experiments/config_MiriGeminiHST_cloudfree.yaml \
       --action evaluate \
       --checkpoint-path /path/to/checkpoints/latest.pt

The evaluator runs:

- **Coverage**: SBC rank statistics
- **Consistency**: Posterior predictive checks with residuals
- **PT Profile**: Temperature-pressure posterior with contribution functions
- **Corner Plot**: Parameter posteriors with derived quantities
- **Bolometric**: T_eff and luminosity from integrated spectra

All outputs are saved to ``evaluations/`` subdirectories with figures as PDFs
and data as CSV/PT files.

Configuration
-------------

Everything is controlled by a single YAML config file. See
``experiments/config_MiriGeminiHST_cloudfree.yaml`` for a complete example.

Key sections:

- ``observation_config``: paths to observed spectra
- ``dataset_config``: simulation dataset paths and pipe configuration
- ``simulator_config``: atmospheric model setup per instrument
- ``prior_config``: parameter bounds
- ``estimator_config``: neural network architecture
- ``training_config``: optimizer, scheduler, loss, epochs
- ``wandb``: logging configuration
