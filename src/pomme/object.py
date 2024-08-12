from astropy             import units
from astropy.coordinates import SkyCoord
from astroquery.simbad   import Simbad


class AstroObject():
    """
    Class to store data about an astrophysical object,
    such as distance, radial velocity, and sky coordinates.
    """

    @units.quantity_input(distance='length', radial_velocity='velocity')
    def __init__(self, name, distance=None, radial_velocity=None):
        """
        Constructor for AstroObject.

        Parameters
        ----------
        name : str
            Name of the astrophysical object.
        distance : astropy.units.quantity.Quantity object, optional
            Distance to the astrophysical object.
        radial_velocity : astropy.units.quantity.Quantity object, optional
            Radial velocity of the astrophysical object.

        Raises
        ------
        RuntimeError
            If the object name is not found in the SIMBAD database.
        """
        # Set name
        self.name = name
        
        # Get coordinates in the sky
        self.coordinates = SkyCoord.from_name(self.name)
        
        # Query SIMBAD for data
        simbadGetter = SimbadGetter(self.name)

        # Set distance
        if distance is not None:
            self.distance = distance
        else:
            self.distance = simbadGetter.get_distance()
            
        # Set radial velocity
        if radial_velocity is not None:
            self.radial_velocity = radial_velocity
        else:
            self.radial_velocity = simbadGetter.get_radial_velocity()

            
class SimbadGetter():
    """
    Class to query data from the SIMBAD data base.
    """
    def __init__(self, name: str):
        """
        Constructor for Simbad getter class.

        Parameters
        ----------
        name : str
            Name of the astrophysical object.

        Raises
        ------
        RuntimeError
            If the object name is not found in the SIMBAD database.
        """
        # Store the object name.
        self.name = name
        
        # Add votable fields to get the relevant data
        Simbad.add_votable_fields('plx', 'distance', 'rv_value')
        
        # Query Simbad based on the name
        self.sim = Simbad.query_object(self.name)
        
        # Check if the query was succesful
        if not self.sim:
            raise RuntimeError("No object with name", self.name, "was found in Simbad.")

            
    def get_distance(self):
        """
        Getter for the distance to the astrophysical object, using Simbad.
        Code adapted from:
        https://gist.github.com/elnjensen/ce2367faf0d876def1ff68b6154e102b

        Returns
        -------
        dist : astropy.units.quantity.Quantity object
            Distance to astrophysical object, from Simbad.
        """
        # Extract parallax
        plx = self.sim['PLX_VALUE']
        # Extract other distance
        dst = self.sim['Distance_distance']
    
        # Prefer the parallax distance if available,
        if not plx.mask[0]:
            print("Using the parallax distance.")
            # Convert parallax to distance
            return plx.to(units.pc, equivalencies=units.parallax())[0]
                      
        # but if no parallax we'll take any other distance.
        elif not dst.mask[0]:
            print("Using the distance from ", self.sim['Distance_bibcode'][0])
            # Extract distance and unit
            return (dst[0] * units.Unit(self.sim['Distance_unit'][0])).to(units.pc)

        # If there is no parallax or distance, we throw an error...
        else: 
            raise RuntimeError("No distance available in Simbad.")
        
        
    def get_radial_velocity(self):
        """
        Getter for the radial velocity of the astrophysical object, using Simbad.

        Returns
        -------
        velo : astropy.units.quantity.Quantity object
             Radial velocity of the astrophysical object, from Simbad.

        Raises
        ------
        RuntimeError
            If the object has no radial velocity in the SIMBAD database.
        """
        # Extract radial velocity
        vel = self.sim['RV_VALUE']
    
        # Check if a value was returned
        if not vel.mask[0]:
            # Convert to SI units
            return vel.to(units.m/units.s)[0]

        # If there is no radial velocity, we throw an error...
        else: 
            raise RuntimeError("No radial velocity available in Simbad.")