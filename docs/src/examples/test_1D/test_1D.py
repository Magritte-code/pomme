import numpy             as np
import torch

from astropy              import units, constants
from pomme.model       import TensorModel, SphericallySymmetric
from pomme.lines       import Line


# Line data
line   = Line('CO', 2)
line_2 = Line('CO', 3)
line_3 = Line('CO', 4)


# Frequency data
vdiff = 300   # velocity increment size [m/s]
nfreq = 100   # number of frequencies
dd    = vdiff / constants.c.si.value * nfreq
fmin  = line.frequency - line.frequency*dd
fmax  = line.frequency + line.frequency*dd

frequencies = torch.linspace(fmin, fmax, nfreq)
velocities  = (frequencies / line.frequency - 1.0) * constants.c.si.value


# Frequency data
vdiff = 300   # velocity increment size [m/s]
nfreq = 100   # number of frequencies
dd    = vdiff / constants.c.si.value * nfreq
fmin  = line_2.frequency - line_2.frequency*dd
fmax  = line_2.frequency + line_2.frequency*dd

frequencies_2 = torch.linspace(fmin, fmax, nfreq)
velocities_2  = (frequencies_2 / line_2.frequency - 1.0) * constants.c.si.value

# Frequency data
vdiff = 300   # velocity increment size [m/s]
nfreq = 100   # number of frequencies
dd    = vdiff / constants.c.si.value * nfreq
fmin  = line_3.frequency - line_3.frequency*dd
fmax  = line_3.frequency + line_3.frequency*dd

frequencies_3 = torch.linspace(fmin, fmax, nfreq)
velocities_3  = (frequencies_3 / line_3.frequency - 1.0) * constants.c.si.value


v_fac = 1.0e-1


# Forward model
def forward(spherical):
    
    spherical.map_1D_to_2D()

    # Extract the projection cosine along the line of sight
    direction = spherical.model_2D.get_radial_direction(origin=spherical.origin_2D)[0]
    direction = direction.nan_to_num()
    
    img = line.LTE_image_along_last_axis(
        density      =         torch.exp(spherical.model_2D['log_CO'          ].T),
        temperature  =         torch.exp(spherical.model_2D['log_temperature' ].T),
        v_turbulence =         torch.exp(spherical.model_2D['log_v_turbulence'].T),
        velocity_los = v_fac * torch.exp(spherical.model_2D['log_velocity'    ].T) * direction.T,
        frequencies  = frequencies,
        dx           = spherical.model_2D.dx(0)
    )

    # Compute the integrated line intensity
    I = spherical.integrate_intensity(img)
    
    return I


# Forward model
def forward_2_lines(spherical):
    
    spherical.map_1D_to_2D()

    # Extract the projection cosine along the line of sight
    direction = spherical.model_2D.get_radial_direction(origin=spherical.origin_2D)[0]
    direction = direction.nan_to_num()
    
    img = line.LTE_image_along_last_axis(
        density      =         torch.exp(spherical.model_2D['log_CO'          ].T),
        temperature  =         torch.exp(spherical.model_2D['log_temperature' ].T),
        v_turbulence =         torch.exp(spherical.model_2D['log_v_turbulence'].T),
        velocity_los = v_fac * torch.exp(spherical.model_2D['log_velocity'    ].T) * direction.T,
        frequencies  = frequencies,
        dx           = spherical.model_2D.dx(0)
    )
    
    img_2 = line_2.LTE_image_along_last_axis(
        density      =         torch.exp(spherical.model_2D['log_CO'          ].T),
        temperature  =         torch.exp(spherical.model_2D['log_temperature' ].T),
        v_turbulence =         torch.exp(spherical.model_2D['log_v_turbulence'].T),
        velocity_los = v_fac * torch.exp(spherical.model_2D['log_velocity'    ].T) * direction.T,
        frequencies  = frequencies_2,
        dx           = spherical.model_2D.dx(0)
    )

    # Compute the integrated line intensity
    I   = spherical.integrate_intensity(img)
    I_2 = spherical.integrate_intensity(img_2)
    
    return I, I_2


# Forward model
def forward_N_lines(spherical, lines, frequencies):
    
    spherical.map_1D_to_2D()

    # Extract the projection cosine along the line of sight
    direction = spherical.model_2D.get_radial_direction(origin=spherical.origin_2D)[0]
    direction = direction.nan_to_num()
    
    imgs = [
             line.LTE_image_along_last_axis(
                 density      =         torch.exp(spherical.model_2D['log_CO'          ].T),
                 temperature  =         torch.exp(spherical.model_2D['log_temperature' ].T),
                 v_turbulence =         torch.exp(spherical.model_2D['log_v_turbulence'].T),
                 velocity_los = v_fac * torch.exp(spherical.model_2D['log_velocity'    ].T) * direction.T,
                 frequencies  = freq,
                 dx           = spherical.model_2D.dx(0)
              )
              for (line, freq) in zip(lines, frequencies)
            ]
    
    # Compute the integrated line intensity
    Is = [spherical.integrate_intensity(img) for img in imgs]

    return Is




def analytic_velo(r, v_in, v_inf, beta):
    return v_in + (v_inf - v_in) * (1.0 - r_in / r)**beta

def analytic_T(r, T_in, epsilon):
    return T_in * (r_in / r)**epsilon


# Forward model
def forward_analytic_velo(spherical):

    r = spherical.model_1D.get_radius(origin=np.array([0]))
    r[r<r_in] = r_in

    velocity = analytic_velo(
        r     = torch.from_numpy(r),
        v_in  = torch.exp(spherical.model_1D['log_v_in']),
        v_inf = torch.exp(spherical.model_1D['log_v_inf']),
        beta  = torch.exp(spherical.model_1D['log_beta'])
    )
    
    spherical.map_1D_to_2D()

    # Extract the projection cosine along the line of sight
    direction = spherical.model_2D.get_radial_direction(origin=spherical.origin_2D)[0]
    direction = direction.nan_to_num()
    
    img = line.LTE_image_along_last_axis(
        density      = torch.exp(spherical.model_2D['log_CO'          ].T),
        temperature  = torch.exp(spherical.model_2D['log_temperature' ].T),
        v_turbulence = torch.exp(spherical.model_2D['log_v_turbulence'].T),
        velocity_los = velocity * direction.T,
        frequencies  = frequencies,
        dx           = spherical.model_2D.dx(0)
    )

    # Compute the integrated line intensity
    I = spherical.integrate_intensity(img)
    
    return I


def forward_analytic_velo_and_T(spherical, line, frequencies):

    r = spherical.model_2D.get_radius(origin=spherical.origin_2D)
    r[r<r_in] = r_in

    velocity = analytic_velo(
        r     = torch.from_numpy(r),
        v_in  = torch.exp(spherical.model_1D['log_v_in']),
        v_inf = torch.exp(spherical.model_1D['log_v_inf']),
        beta  = torch.exp(spherical.model_1D['log_beta'])
    )
    
    temperature = analytic_T(
        r       = torch.from_numpy(r),
        T_in    = torch.exp(spherical.model_1D['log_T_in']),
        epsilon = torch.exp(spherical.model_1D['log_epsilon'])
    )

    spherical.map_1D_to_2D()

    # Extract the projection cosine along the line of sight
    direction = spherical.model_2D.get_radial_direction(origin=spherical.origin_2D)[0]
    direction = direction.nan_to_num()
    
    img = line.LTE_image_along_last_axis(
        density      = torch.exp(spherical.model_2D['log_CO'          ].T),
        temperature  = temperature.T,
        v_turbulence = torch.exp(spherical.model_2D['log_v_turbulence'].T),
        velocity_los = velocity.T * direction.T,
        frequencies  = frequencies,
        dx           = spherical.model_2D.dx(0)
    )

    # Compute the integrated line intensity
    I = spherical.integrate_intensity(img)
    
    return I


def forward_analytic_velo_and_T_2_lines(spherical):

    r = spherical.model_2D.get_radius(origin=spherical.origin_2D)
    r[r<r_in] = r_in

    velocity = analytic_velo(
        r     = torch.from_numpy(r),
        v_in  = torch.exp(spherical.model_1D['log_v_in']),
        v_inf = torch.exp(spherical.model_1D['log_v_inf']),
        beta  = torch.exp(spherical.model_1D['log_beta'])
    )
    
    temperature = analytic_T(
        r       = torch.from_numpy(r),
        T_in    = torch.exp(spherical.model_1D['log_T_in']),
        epsilon = torch.exp(spherical.model_1D['log_epsilon'])
    )

    spherical.map_1D_to_2D()

    # Extract the projection cosine along the line of sight
    direction = spherical.model_2D.get_radial_direction(origin=spherical.origin_2D)[0]
    direction = direction.nan_to_num()
    
    img = line.LTE_image_along_last_axis(
        density      = torch.exp(spherical.model_2D['log_CO'          ].T),
        temperature  = temperature.T,
        v_turbulence = torch.exp(spherical.model_2D['log_v_turbulence'].T),
        velocity_los = velocity.T * direction.T,
        frequencies  = frequencies,
        dx           = spherical.model_2D.dx(0)
    )
    
    img_2 = line_2.LTE_image_along_last_axis(
        density      = torch.exp(spherical.model_2D['log_CO'          ].T),
        temperature  = temperature.T,
        v_turbulence = torch.exp(spherical.model_2D['log_v_turbulence'].T),
        velocity_los = velocity.T * direction.T,
        frequencies  = frequencies_2,
        dx           = spherical.model_2D.dx(0)
    )

    # Compute the integrated line intensity
    I   = spherical.integrate_intensity(img)
    I_2 = spherical.integrate_intensity(img_2)
    
    return I, I_2


def forward_analytic_velo_and_T_N_lines(spherical, lines, frequencies):

    r = spherical.model_2D.get_radius(origin=spherical.origin_2D)
    r[r<r_in] = r_in

    velocity = analytic_velo(
        r     = torch.from_numpy(r),
        v_in  = torch.exp(spherical.model_1D['log_v_in']),
        v_inf = torch.exp(spherical.model_1D['log_v_inf']),
        beta  = torch.exp(spherical.model_1D['log_beta'])
    )
    
    temperature = analytic_T(
        r       = torch.from_numpy(r),
        T_in    = torch.exp(spherical.model_1D['log_T_in']),
        epsilon = torch.exp(spherical.model_1D['log_epsilon'])
    )

    spherical.map_1D_to_2D()

    # Extract the projection cosine along the line of sight
    direction = spherical.model_2D.get_radial_direction(origin=spherical.origin_2D)[0]
    direction = direction.nan_to_num()
    
    temperature[r>r_out] = 0.1
    spherical.model_2D['log_CO'][r>r_out] = -100.0

    imgs = [
             line.LTE_image_along_last_axis(
                 density      = torch.exp(spherical.model_2D['log_CO'          ].T),
                 temperature  = temperature.T,
                 v_turbulence = torch.exp(spherical.model_2D['log_v_turbulence'].T),
                 velocity_los = velocity.T * direction.T,
                 frequencies  = freq,
                 dx           = spherical.model_2D.dx(0)
              )
              for (line, freq) in zip(lines, frequencies)
            ]
    
    # Compute the integrated line intensity
    Is = [spherical.integrate_intensity(img) for img in imgs]
    
    return Is, imgs




n_elements = 128

r_in  = (1.0e+2 * units.au).si.value
r_out = (1.0e+4 * units.au).si.value

v_in  = (1.0e+0 * units.km / units.s).si.value
v_inf = (2.0e+1 * units.km / units.s).si.value
beta  = 0.5

T_in    = (1.0e+3 * units.K).si.value
epsilon = 0.3

Mdot = (1.0e-6 * units.M_sun / units.yr).si.value

v    = lambda r: v_in + (v_inf - v_in) * (1.0 - r_in / r)**beta
T    = lambda r: T_in * (r_in / r)**epsilon
rho  = lambda r: Mdot / (4.0 * np.pi * r**2 * v(r))
n_CO = lambda r: 1.0e-4 * constants.N_A.si.value / 2.02 * rho(r)

v_turb = (0.25 * units.km / units.s).si.value



def get_model():

    model_1D = TensorModel(sizes=r_out, shape=n_elements)

    r = model_1D.get_radius(origin=np.array([0]))
    r[r<r_in] = r_in

    # Define and initialise the model variables
    model_1D['log_velocity'    ] = np.log(1/v_fac * v(r))
    model_1D['log_CO'          ] = np.log(n_CO(r))
    model_1D['log_temperature' ] = np.log(   T(r))
    model_1D['log_v_turbulence'] = np.log(v_turb) * np.ones_like(r)

    # Create a spherically symmetric model form the 1D TensorModel
    spherical = SphericallySymmetric(model_1D)

    return spherical


def get_obs():
    return forward(get_model())