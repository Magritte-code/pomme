import torch
import numpy             as np
import matplotlib.pyplot as plt

from astropy    import units, constants
from torch.nn   import functional
from ipywidgets import interact

# Cosmic Microwave Background temperature
T_CMB = 2.725


def planck(temperature, frequency):
    '''
    Planck function for thermal radiation.

    Parameters
    ----------
    temperature : float
        Temperature at which to evaluate the intensity.
    frequency : float
        Frequency at which to evaluate the intensity.

    Returns
    -------
    out : float
        Planck function evaluated at the frequency for the given temperature.
    '''
    # Specify constants
    h  = constants.h  .si.value
    c  = constants.c  .si.value
    kb = constants.k_B.si.value
    # Return planck function
    return 2.0*h/c**2 * np.power(frequency, 3) / np.expm1(h*frequency/(kb*temperature))


def print_var(name, var):
    """
    Print the min, mean, and max of a PyTorch tensor variable.

    Parameters
    ----------
    name : str
        Name of the variable.
    var : torch.Tensor
        Variable to print the min, mean, and max of.

    Returns
    -------
    out : None
    """
    print(f"{name} {var.min().item():+1.2e} {var.mean().item():+1.2e} {var.max().item():+1.2e}")


def interpolate(inp, size, mode='nearest'):
    """
    Interpolate a tensor to a new size.

    Parameters
    ----------
    inp : torch.Tensor
        Input tensor to interpolate.
    size : tuple
        New size of the tensor.
    mode : str
        Interpolation mode.
    
    Returns
    -------
    out : torch.Tensor
        Interpolated tensor.
    """
    # Reshape to add batch ids and channels
    # (Because that is what torch.nn.functional.interpolate expects.)
    res = inp.view((1, 1) + inp.size())
    # Interpolate
    res = functional.interpolate(res, size=size, mode=mode)
    # Return without batch ids and channels
    return res.view(size)


@units.quantity_input(angle='angle', distance='length')
def convert_angular_to_spatial(angle, distance):
    """
    Convert angles to distances assuming a certain distance.

    Parameters
    ----------
    angle : astropy.units.Quantity
        Angle to convert.
    distance : astropy.units.Quantity
        Distance to assume.

    Returns
    -------
    out : astropy.units.Quantity
        Distance corresponding to the angle at the given distance.
    """
    angle    = angle   .to(units.arcsec).value
    distance = distance.to(units.pc    ).value
    return angle * distance * units.au


# https://gist.github.com/elibroftw/22e3b4c1eb7fa0a6c83d099d24200f95
# =================================
# Molar Mass Calculator
# Author: Elijah Lopez
# Version: 1.2.1
# Last Updated: April 4th 2020
# Created: July 8th 2017
# Python Version: 3.6+
# =================================
MM_of_Elements = {
    'H': 1.00794, 'He': 4.002602, 'Li': 6.941, 'Be': 9.012182, 'B': 10.811, 'C': 12.0107, 'N': 14.0067,
    'O': 15.9994, 'F': 18.9984032, 'Ne': 20.1797, 'Na': 22.98976928, 'Mg': 24.305, 'Al': 26.9815386,
    'Si': 28.0855, 'P': 30.973762, 'S': 32.065, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.0983, 'Ca': 40.078,
    'Sc': 44.955912, 'Ti': 47.867, 'V': 50.9415, 'Cr': 51.9961, 'Mn': 54.938045,
    'Fe': 55.845, 'Co': 58.933195, 'Ni': 58.6934, 'Cu': 63.546, 'Zn': 65.409, 'Ga': 69.723, 'Ge': 72.64,
    'As': 74.9216, 'Se': 78.96, 'Br': 79.904, 'Kr': 83.798, 'Rb': 85.4678, 'Sr': 87.62, 'Y': 88.90585,
    'Zr': 91.224, 'Nb': 92.90638, 'Mo': 95.94, 'Tc': 98.9063, 'Ru': 101.07, 'Rh': 102.9055, 'Pd': 106.42,
    'Ag': 107.8682, 'Cd': 112.411, 'In': 114.818, 'Sn': 118.71, 'Sb': 121.760, 'Te': 127.6,
    'I': 126.90447, 'Xe': 131.293, 'Cs': 132.9054519, 'Ba': 137.327, 'La': 138.90547, 'Ce': 140.116,
    'Pr': 140.90465, 'Nd': 144.242, 'Pm': 146.9151, 'Sm': 150.36, 'Eu': 151.964, 'Gd': 157.25,
    'Tb': 158.92535, 'Dy': 162.5, 'Ho': 164.93032, 'Er': 167.259, 'Tm': 168.93421, 'Yb': 173.04,
    'Lu': 174.967, 'Hf': 178.49, 'Ta': 180.9479, 'W': 183.84, 'Re': 186.207, 'Os': 190.23, 'Ir': 192.217,
    'Pt': 195.084, 'Au': 196.966569, 'Hg': 200.59, 'Tl': 204.3833, 'Pb': 207.2, 'Bi': 208.9804,
    'Po': 208.9824, 'At': 209.9871, 'Rn': 222.0176, 'Fr': 223.0197, 'Ra': 226.0254, 'Ac': 227.0278,
    'Th': 232.03806, 'Pa': 231.03588, 'U': 238.02891, 'Np': 237.0482, 'Pu': 244.0642, 'Am': 243.0614,
    'Cm': 247.0703, 'Bk': 247.0703, 'Cf': 251.0796, 'Es': 252.0829, 'Fm': 257.0951, 'Md': 258.0951,
    'No': 259.1009, 'Lr': 262, 'Rf': 267, 'Db': 268, 'Sg': 271, 'Bh': 270, 'Hs': 269, 'Mt': 278,
    'Ds': 281, 'Rg': 281, 'Cn': 285, 'Nh': 284, 'Fl': 289, 'Mc': 289, 'Lv': 292, 'Ts': 294, 'Og': 294,
    '': 0
}


def get_molar_mass(compound: str, decimal_places=None) -> float:
    """
    Compute the molar mass based on a chemical formula.
    Adapted from https://gist.github.com/elibroftw/22e3b4c1eb7fa0a6c83d099d24200f95

    Parameters
    ----------
    compound : str
        Chemical formula to compute the molar mass of.
    decimal_places : int
        Number of decimal places to round the molar mass to.
    
    Returns
    -------
    out : float
        Molar mass of the compound.
    """
    is_polyatomic = end = multiply = False
    polyatomic_mass, m_m, multiplier = 0, 0, 1
    element = ''

    for e in compound:
        if is_polyatomic:
            if end:
                is_polyatomic = False
                m_m += int(e) * polyatomic_mass if e.isdigit() else polyatomic_mass + MM_of_Elements[e]
            elif e.isdigit():
                multiplier = int(str(multiplier) + e) if multiply else int(e)
                multiply = True
            elif e.islower():
                element += e
            elif e.isupper():
                polyatomic_mass += multiplier * MM_of_Elements[element]
                element, multiplier, multiply = e, 1, False
            elif e == ')':
                polyatomic_mass += multiplier * MM_of_Elements[element]
                element, multiplier = '', 1
                end, multiply = True, False
        elif e == '(':
            m_m += multiplier * MM_of_Elements[element]
            element, multiplier = '', 1
            is_polyatomic, multiply = True, False
        elif e.isdigit():
            multiplier = int(str(multiplier) + e) if multiply else int(e)
            multiply = True
        elif e.islower():
            element += e
        elif e.isupper():
            m_m += multiplier * MM_of_Elements[element]
            element, multiplier, multiply = e, 1, False
    m_m += multiplier * MM_of_Elements[element]
    if decimal_places is not None:
        return round(m_m, decimal_places)
    return m_m