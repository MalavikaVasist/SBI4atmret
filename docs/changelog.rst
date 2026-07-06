Changelog
=========

v0.1.0 (2025-07)
-----------------

Initial release.

- Config-driven workflow (YAML → Pydantic validation)
- Dataset generation script (``generate_dataset.py``)
- Training with NPE/BNPE loss, wandb logging
- Evaluation suite:
  - Coverage (SBC ranks)
  - Posterior predictive consistency checks
  - PT profile posterior with contribution functions
  - Corner plots with derived parameters
  - Bolometric (T_eff, luminosity)
  - Importance sampling refinement
  - Bayes factor via learned classifiers
  - OOD detection (posterior variability, embedding PCA)
- Meta-learner ensemble for posterior stacking
- Multi-instrument support (MIRI, HST, Gemini)
- Heteroscedastic noise model with b-factor
