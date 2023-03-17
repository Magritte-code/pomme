import numpy             as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy    import units
from ipywidgets import interact


class Observation:
    
    def __init__(self):
        
        print('Created an Observation object!')
        

class DataCube():
    """
    Data structure for a data cube of spectral line images.
    """
    
    def __init__(self, fits_file):
        """
        Constructor for DataCube.
        """
        self.read_fits_file(fits_file)
    
    
    def read_fits_file(self, fits_file):
        """
        Reader for fits files containing spectral line images.
        """
        #Load the data
        with fits.open(fits_file) as hdu:
            # Extract data
            self.img = hdu[0].data[0].astype(np.float32)   # Store as 32 bit numbers to avoid too much memory
            # Extract header
            self.hdr = hdu[0].header                       

            # Extract the object name
            self.object = str(self.hdr['OBJECT']).strip()
            
            # Extract units for intensities
            self.I_unit = units.Unit(self.hdr['BUNIT'])
            
            # Extract units for axis
            self.x_unit = units.Unit(self.hdr['CUNIT1'])
            self.y_unit = units.Unit(self.hdr['CUNIT2'])
            self.f_unit = units.Unit(self.hdr['CUNIT3'])
            
            # Extract data from the header
            self.npix_x    =   int(self.hdr['NAXIS1'])
            self.npix_y    =   int(self.hdr['NAXIS2'])
            self.npix_f    =   int(self.hdr['NAXIS3'])

            self.pixsize_x = float(self.hdr['CDELT1']) * self.x_unit
            self.pixsize_y = float(self.hdr['CDELT2']) * self.y_unit
            self.pixsize_f = float(self.hdr['CDELT3']) * self.f_unit

            self.refpix_x  =   int(self.hdr['CRPIX1']) - 1
            self.refpix_y  =   int(self.hdr['CRPIX2']) - 1
            self.refpix_f  =   int(self.hdr['CRPIX3']) - 1

            self.refval_x  = float(self.hdr['CRVAL1']) * self.x_unit
            self.refval_y  = float(self.hdr['CRVAL2']) * self.y_unit
            self.refval_f  = float(self.hdr['CRVAL3']) * self.f_unit

            self.restfreq  = float(self.hdr['RESTFRQ'])

            # Verify image axis
            if (self.img.shape[1] != self.npix_x):
                raise ValueError('Number of pixels mismatch (AXIS 1, x).')
            if (self.img.shape[2] != self.npix_y):
                raise ValueError('Number of pixels mismatch (AXIS 2, y).')
            if (self.img.shape[0] != self.npix_f):
                raise ValueError('Number of pixels mismatch (AXIS 0, v).')
            if (self.hdr['CTYPE1'] != 'RA---SIN'):
                raise NotImplementedError("Don't know how to handle this.")
            if (self.hdr['CTYPE2'] != 'DEC--SIN'):
                raise NotImplementedError("Don't know how to handle this.")
            
            # Construct image axis
            self.xs = self.pixsize_x * (np.arange(self.npix_x) - self.refpix_x) + self.refval_x
            self.ys = self.pixsize_y * (np.arange(self.npix_y) - self.refpix_y) + self.refval_y
            self.fs = self.pixsize_f * (np.arange(self.npix_f) - self.refpix_f) + self.refval_f
            
            # Extract ranges for intensities
            self.img_min, self.img_max = np.min(self.img), np.max(self.img)
            
            # Extract ranges for axis
            self.x_min, self.x_max = np.min(self.xs), np.max(self.xs)
            self.y_min, self.y_max = np.min(self.ys), np.max(self.ys)
            self.f_min, self.f_max = np.min(self.fs), np.max(self.fs)
            

    def plot_channel_maps(self):
        """
        Plot channel maps for this observation.
        """
        # Define a plot function
        def plot(f):
            plt.figure(dpi=150)
            plt.imshow(self.img[f,:,:], vmin=self.img_min, vmax=self.img_max)
            
        # Return an interactive ipywidget
        return interact(plot, f=(0, self.img.shape[0]-1))
    
    
    def plot_xy_maps(self):
        """
        Plot x / y maps for this observation.
        (Alias for plot_channel_maps.)
        """
        return self.plot_channel_maps()
    
    
    def plot_xf_maps(self):
        """
        Plot x-axis / frequency maps for this observation.
        """
        # Define a plot function
        def plot(i):
            plt.figure(dpi=150)
            plt.imshow(self.img[:,i,:], vmin=self.img_min, vmax=self.img_max)
            
        # Return an interactive ipywidget
        return interact(plot, i=(0, self.img.shape[1]-1))
    
    
    def plot_yf_maps(self):
        """
        Plot y-axis / frequency maps for this observation.
        """
        # Define a plot function
        def plot(j):
            plt.figure(dpi=150)
            plt.imshow(self.img[:,:,j], vmin=self.img_min, vmax=self.img_max)
            
        # Return an interactive ipywidget
        return interact(plot, j=(0, self.img.shape[2]-1))
    
    
    def plot_spectrum(self):
        """
        Plot spectrum at a pixel for this observation.
        """
        # Define a plot function
        def plot(i,j):
            plt.figure(dpi=150)
            plt.step(self.fs, self.img[:,i,j])
            plt.ylim((self.img_min, self.img_max))
            plt.xlabel(f'frequency [{self.f_unit}]')
            plt.ylabel(f'intensity [{self.I_unit}]')

        # Return an interactive ipywidget
        return interact(plot,
                        i=(0, self.img.shape[1]-1),
                        j=(0, self.img.shape[2]-1) )