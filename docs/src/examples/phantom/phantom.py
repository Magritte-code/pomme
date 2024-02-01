import torch

from astropy        import constants
from pomme.lines import Line
from pomme.utils import get_molar_mass, planck, T_CMB


# Lines
lines = [
    Line(species_name='CO',     transition=3),
    Line(species_name='sio-h2', transition=2, molar_mass=get_molar_mass('SiO'))
]

# Molecular fractions
fracs = [3.0e-4, 5.0e-6]

# Frequency / velocity range
vdiff = 120   # velocity increment size [m/s]
nfreq = 100   # number of frequencies

velos  = nfreq * vdiff * torch.linspace(-1, +1, nfreq, dtype=torch.float64)


class Model:

    def __init__(self, model, lines, fracs, velos):        
        self.model = model
        self.lines = lines
        self.fracs = fracs
        self.velos = velos
        self.freqs = [(1.0 + self.velos / constants.c.si.value) * line.frequency for line in lines]


    def get_velocity(self, model):
        raise NotImplementedError('First implement and define get_velocity.')


    def get_temperature(self, model):
        raise NotImplementedError('First implement and define get_temperature.')


    def get_abundance(self, model):
        raise NotImplementedError('First implement and define get_abundance.')


    def get_turbulence(self, model):
        raise NotImplementedError('First implement and define get_turbulence.')


    def get_boundary_condition(self, model, frequency):
        raise NotImplementedError('First implement and define get_boundary_condition.')


    # Forward model
    def image(self):

        # Tensor for the intensities in each line
        imgs = torch.zeros((len(lines), self.model.shape[0], self.model.shape[1], len(self.freqs[0])), dtype=torch.float64)

        # For each line
        for l, (line, frac, freq) in enumerate(zip(self.lines, self.fracs, self.freqs)):

            # Check that the number of frequencies is the same for all lines
            assert len(freq) == imgs[l].shape[-1]

            imgs[l] = line.LTE_image_along_last_axis(
                abundance    = self.get_abundance  (self.model) * frac,
                temperature  = self.get_temperature(self.model),
                v_turbulence = self.get_turbulence (self.model),
                velocity_los = self.get_velocity   (self.model)[2],
                frequencies  = freq,
                dx           = self.model.dx(0),
                img_bdy      = self.get_boundary_condition(self.model, freq)
            )

        return imgs