Installation
============

Requirements
------------

- Python >= 3.10
- PyTorch >= 2.0
- petitRADTRANS (for atmospheric simulation)
- lampe (for neural posterior estimation)
- zuko (normalizing flow distributions)

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/mvasist/SBI4atmret.git
   cd SBI4atmret
   pip install -e .

Dependencies
------------

Core dependencies are listed in ``pyproject.toml``. Key packages:

.. code-block:: text

   torch>=2.0
   lampe>=0.8
   zuko>=1.0
   petitRADTRANS>=2.4
   pydantic>=2.0
   numpy
   scipy
   matplotlib
   pandas
   h5py
   wandb
   tqdm
   astropy

Optional (for documentation):

.. code-block:: bash

   pip install -r docs/requirements.txt
