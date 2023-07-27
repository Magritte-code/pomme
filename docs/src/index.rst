p3droslo
########

Welcome to the *p3droslo* documentation!

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   background/index
   examples/index
   API/index


About
*****

*p3droslo* is a library for **probabilistic 3D reconstruction** of astronomical **spectral line observations**.

Observations of `spectral lines <https://en.wikipedia.org/wiki/Spectral_line>`_ are indespensible in astronomy, since they encode a wealth of information about the physical and chemical conditions of the medium in which they were formed.
For instance, their narrow extent in frequency space make them very sensitive to Doppler shifts, such that their shape encodes the motion of the medium along the line of sight.
As a result, given a good model for the line formation process and a good inversion method, these physical and chemical properties can be retrieved from observations.
Currently, we mainly focus on retrieving the distributions of the abundance of the chemical species producing the line, the velocity field, and its kinetic temperature.

More information about the model for `spectral line formation <https://p3droslo.readthedocs.io/en/latest/background/spectral_line_formation.html>`_ and the `probabilistic reconstruction methods <https://p3droslo.readthedocs.io/en/latest/background/probabilistic_reconstruction.html>`_ can be found in the `background <https://p3droslo.readthedocs.io/en/latest/background/index.html>`_ pages.

*p3droslo* is built on top of `PyTorch <https://pytorch.org>`_ and benefits a lot from functionality provided by `Astropy <https://www.astropy.org>`_.
It is currently developed and maintained by `Frederik De Ceuster <https://freddeceuster.github.io>`_ at `KU Leuven <https://www.kuleuven.be/english/kuleuven/index.html>`_.


Installation
************

Get the latest release (version 0.0.14) either from `PyPI <https://pypi.org/project/p3droslo/>`_, using pip, with:

.. code-block:: shell

    pip install p3droslo

or from `Anaconda.org <https://anaconda.org/FredDeCeuster/p3droslo>`_, using conda, with:

.. code-block:: shell

    conda install -c freddeceuster p3droslo 

or download the `source code <https://github.com/Magritte-code/p3droslo/archive/refs/heads/main.zip>`_, unzip, and install with pip by executing:

.. code-block:: shell

    pip install .

in the root directory of the code.


Issues
******

Please report any issues with this software or its documentation `here <https://github.com/Magritte-code/p3droslo/issues>`_.


Contributing
************

We are open to contributions to p3droslo. More information can be found `here <https://github.com/Magritte-code/p3droslo/blob/main/CONTRIBUTING.md>`_.


Collaborating
*************

We are always interested in collaborating! If you have a project, or idea, or data that might benefit from probabilistic 3D reconstruction or any other method you can find in this repository, feel free to contact `me <https://freddeceuster.github.io>`_.


Acknowledgements
****************

Frederik De Ceuster is a Postdoctoral Research Fellow of the `Research Foundation - Flanders (FWO) <https://www.fwo.be/en/>`_, grant number 1253223N, and was previously supported for this research by a Postdoctoral Mandate (PDM) from `KU Leuven <https://www.kuleuven.be/english/kuleuven/index.html>`_, grant number PDMT2/21/066.


|

.. image:: images/FWO_logo.jpeg
  :width: 55%
  :align: right
  :alt: FWO logo
.. image:: images/KU_Leuven_logo.png
  :width: 30%
  :align: left
  :alt: KU Leuven logo