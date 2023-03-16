from astropy           import units
from astroquery.simbad import Simbad


Simbad.add_votable_fields('plx', 'distance')


def get_distance_from_Simbad(name):
    """
    Getter for the distance to the astrophysical object, using Simbad.

    Parameters
    ----------
    name : str
        Name of the object.

    Returns
        -------
    dist : astropy.units.quantity.Quantity object
        Distance to astrophysical object, from Simbad.

    Code adapted from:
    https://gist.github.com/elnjensen/ce2367faf0d876def1ff68b6154e102b
    """
    # Query Simbad based on the name
    sim = Simbad.query_object(name)
    
    # Check if the query was succesful
    if not sim:
        raise RuntimeError("No object with name", name, "was found in Simbad.")
    
    # Extract parallax
    plx = sim['PLX_VALUE']
    # Extract other distance
    dst = sim['Distance_distance']
    
    # Prefer the parallax distance if available,
    if not plx.mask[0]:
        print("Using the parallax distance.")
        # Convert parallax to distance
        return plx.to(units.pc, equivalencies=units.parallax())[0]
                      
    # but if no parallax we'll take any other distance.
    elif not dst.mask[0]:
        print("Using the distance from ", sim['Distance_bibcode'][0])
        # Extract distance and unit
        return (dst[0] * units.Unit(sim['Distance_unit'][0])).to(units.pc)

    # If there is no parallax or distance, we throw an error...
    else: 
        raise RuntimeError("No distance available in Simbad.")


@units.quantity_input(angle='angle', distance='length')
def convert_angular_to_spatial(angle, distance):
    """
    Convert angles to distances assuming a certain distance.
    """
    angle    = angle   .to(units.arcsec).value
    distance = distance.to(units.pc    ).value
    return angle * distance * units.au