Examples
========

Running Individual Evaluations
------------------------------

After training, you can run evaluations individually:

.. code-block:: python

   from sbi4atmret.evaluation.EvaluateBase import BaseEvaluator

   # ... setup evaluator ...

   # Coverage only
   result = evaluator.run_coverage()
   print(f"Coverage plot saved")

   # Corner plot with specific parameters
   result = evaluator.run_corner(
       param_names_to_plot=["$R_P$", "$Mass$", "$H_2O$", "$NH_3$", "$^{15}NH_3$"],
       derived_params=[
           {"name": r"$^{14}N/^{15}N$", "fn": ratio_14N_15N, "lower": 0, "upper": 1000},
       ],
   )

   # PT profile with contribution
   result = evaluator.run_PT(show_contribution=True, xlim=(0, 4000))

   # Bolometric properties
   result = evaluator.run_bolometric(distance=7.34)
   print(f"T_eff = {result.teff.mean():.0f} K")

OOD Detection
-------------

.. code-block:: python

   from sbi4atmret.evaluation.ood_tests import compute_ood_score, plot_variability
   from sbi4atmret.models.meta_learner import load_base_models

   # Load multiple trained models
   base_models = load_base_models(checkpoint_paths, model_builder=...)

   # Posterior variability score
   result = compute_ood_score(base_models, x_obs, n_samples=2048)
   print(f"D_v = {result.variability_score:.4f}")
   fig = plot_variability(result)

Bayes Factor
------------

.. code-block:: python

   from sbi4atmret.evaluation.bayes_factor import (
       BayesFactorClassifier, prepare_classification_data,
       train_bayes_classifier, compute_bayes_factor,
   )

   # Prepare data from two competing models
   spectra_dict = {"cloudfree": spectra_cf, "cloudy": spectra_cl}
   x, labels, model_names = prepare_classification_data(spectra_dict)

   # Train classifier (uses frozen NPE embedding)
   classifier = BayesFactorClassifier(embedding=model.estimator.embedding, ...)
   history = train_bayes_classifier(classifier, train_x, train_labels, ...)

   # Evaluate on observation
   result = compute_bayes_factor(classifier, x_obs, model_names)
   print(result.interpretation)

Importance Sampling
-------------------

.. code-block:: python

   from sbi4atmret.evaluation.importance_sampling import ImportanceSampler

   # IS refinement of the posterior
   is_sampler = ImportanceSampler.__new__(ImportanceSampler)
   is_sampler.__dict__.update(evaluator.__dict__)

   result = is_sampler.run(n_samples=10000, batch_size=500)
   print(f"ESS: {result.n_eff:.0f}, log Z: {result.log_evidence:.2f}")

   # Then plot IS-weighted corner
   evaluator.IS_corner()
