# pomme

_Probabilistic 3D Reconstruction of Spectral Line Observations._


![Build status](https://github.com/Magritte-code/pomme/actions/workflows/build-and-test.yaml/badge.svg)
![Build status](https://github.com/Magritte-code/pomme/actions/workflows/upload-to-pypi.yaml/badge.svg)
![Build status](https://github.com/Magritte-code/pomme/actions/workflows/upload-to-anaconda.yaml/badge.svg)


## About

*pomme* is a python package that allows you to create **probabilistic 3D reconstructions** of astronomical **spectral line observations**.

Observations of [spectral lines](https://en.wikipedia.org/wiki/Spectral_line) are indespensible in astronomy, since they encode a wealth of information about the physical and chemical conditions of the medium from which they originate.
Their narrow extent in frequency space make them very sensitive to Doppler shifts, such that their shape encodes the motion of the medium along the line of sight.
As a result, given a good model for the line formation process and an inversion method, these physical and chemical properties can be retrieved from observations.
Currently, we mainly focus on retrieving the distributions of the abundance of the chemical species producing the line, the velocity field, and its kinetic temperature.
However, also other parameters can be retrieved.

More information about the model, our methods, and their implementation can be found in [De Ceuster et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240218525D/abstract).

*pomme* is built on top of [PyTorch](https://pytorch.org) and benefits a lot from functionality provided by [Astropy](https://www.astropy.org).
It is currently developed and maintained by [Frederik De Ceuster](https://freddeceuster.github.io) at [KU Leuven](https://www.kuleuven.be/english/kuleuven/index.html).


## Installation

Get the latest release (version 0.0.17) either from [PyPI](https://pypi.org/project/pomme/), using `pip`, with:
```
pip install pomme
```
or from [Anaconda.org](https://anaconda.org/FredDeCeuster/pomme), using `conda`, with:
```
conda install -c freddeceuster pomme 
```
or download the [source code](https://github.com/Magritte-code/pomme/archive/refs/heads/main.zip), unzip, and install with `pip` by executing:
```
pip install .
```
in the root directory of the code.


## Documentation

Documentation with examples can be found at [pomme.readthedocs.io](https://pomme.readthedocs.io/en/latest/).


## Issues

Please report any issues with this software or its documentation [here](https://github.com/Magritte-code/pomme/issues).


## Contributing

We are open to contributions to pomme. More information can be found [here](https://github.com/Magritte-code/pomme/blob/main/CONTRIBUTING.md).


## Collaborating

We are always interested in collaborating!
If you like our work but it needs some tailoring for your specific use case feel free to contact [me](https://freddeceuster.github.io).


## Acknowledgements

Frederik De Ceuster is a Postdoctoral Research Fellow of the [Research Foundation - Flanders (FWO)](https://www.fwo.be/en/), grant number 1253223N, and was previously supported for this research by a Postdoctoral Mandate (PDM) from [KU Leuven](https://www.kuleuven.be/english/kuleuven/index.html), grant number PDMT2/21/066.