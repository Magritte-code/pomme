import torch
import numpy as np

from os                import path
from pomme.observation import DataCube, Beam

# Get the path to this folder
this_folder = path.dirname(path.abspath(__file__))


def test_torch_vs_numpy_beam_kernel():
    # Read the data cube
    data = DataCube(f'{this_folder}/data/header_only.fits')
    # Extract the beam
    beam = Beam(data)

    N = 50

    # Create a random image
    img_torch = torch.rand(N,N,1, dtype=torch.float64)
    img_numpy = img_torch[:,:,0].numpy()

    # Convolve once using the astropy functionality and once with the torch implementation
    img_torch_conv = beam.torch_apply(img_torch)[:,:,0].numpy()
    img_numpy_conv = beam.apply      (img_numpy)

    # Compute the absolute relative difference
    err = np.abs(2.0 * (img_torch_conv - img_numpy_conv) / (img_torch_conv + img_numpy_conv))

    assert err.max() < 5.0e-14