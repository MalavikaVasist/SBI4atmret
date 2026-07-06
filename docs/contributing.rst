Contributing
============

Development Setup
-----------------

.. code-block:: bash

   git clone https://github.com/mvasist/SBI4atmret.git
   cd SBI4atmret
   pip install -e ".[dev]"

Branch Workflow
--------------

- Create feature branches from ``main``: ``feature/<description>``
- Bug fixes: ``fix/<description>``
- Never push directly to ``main``
- Open a pull request for review before merging

Code Style
----------

- Follow PEP 8
- Use Google-style docstrings
- Type hints on all public functions
- Keep modules focused on a single responsibility

Testing
-------

.. code-block:: bash

   pytest tests/

Documentation
-------------

Build docs locally:

.. code-block:: bash

   cd docs
   pip install -r requirements.txt
   make html

Open ``_build/html/index.html`` in a browser to preview.

Adding a New Evaluation Method
------------------------------

1. Create a module in ``sbi4atmret/evaluation/`` (e.g., ``my_eval.py``)
2. Define a ``Result`` dataclass and an ``Evaluator`` class
3. Add a ``run_my_eval()`` method to ``BaseEvaluator`` in ``EvaluateBase.py``
4. Follow the pattern: check cache → compute → save → plot → return result
