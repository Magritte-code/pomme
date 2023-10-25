import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
import numpy as np
import torch

from p3droslo.model import TensorModel, SphericallySymmetric
from p3droslo.loss import Loss

from test_1D                  import get_model, get_obs, forward, frequencies, velocities, r_in, v_fac
from test_1D_CO_and_T_and_vel import get_initial_model, reconstruct, reconstruct2

test_model_name = 'models/test_model_1D.h5'
spherical_model = get_model()
spherical_model.model_1D.save(test_model_name)

obs = get_obs()

model_1D = TensorModel.load('models/model_all.h5')

img, loss = reconstruct2(SphericallySymmetric(model_1D), obs)

model_1D.save('models/model_all.h5')