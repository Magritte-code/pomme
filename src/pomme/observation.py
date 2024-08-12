import torch
import numpy             as np
import matplotlib.pyplot as plt

from astropy.io          import fits
from astropy             import units
from ipywidgets          import interact
from radio_beam          import Beam as RadioBeam
from astropy.convolution import convolve


# class Observation:    
#     def __init__(self):
#         print('Created an Observation object!')
        

class DataCube():
    """
    Data structure for a data cube of spectral line images.
    Helps to conveniently read and visualize spectral line images.
    """
    
    def __init__(self, fits_file):
        """
        Constructor for DataCube.

        Parameters
        ----------
        fits_file : str
            Path to the fits file containing the spectral line image.
        """
        self.read_fits_file(fits_file)
    
    
    def read_fits_file(self, fits_file):
        """
        Reader for fits files containing spectral line images.

        Parameters
        ----------
        fits_file : str
            Path to the fits file containing the spectral line image.
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


class Beam():
    
    def __init__(self, datacube):
        """
        Constructor for Beam.

        Parameters
        ----------
        datacube : DataCube
            DataCube object containing the spectral line image.
        """
        if abs(datacube.pixsize_x) != abs(datacube.pixsize_y):
            raise ValueError("Pixels are not square! Cannot handle non-square pixels!")

        # Create a radio-beam Beam object        
        self.object = RadioBeam.from_fits_header(datacube.hdr)

        # Extract the beam as a kernel
        self.kernel = self.object.as_kernel(datacube.pixsize_x)

        # Extract the beam as a torch kernel
        self.torch_kernel = self.get_torch_kernel(datacube.pixsize_x)


    def get_torch_kernel(self, pixsize):
        """
        Getter for the beam of the observation as a torch kernel.
        
        Parameters
        ----------
        pixsize : float
            Pixel size in the same units as the beam.

        Returns
        -------
        torch_kernel : torch.nn.Conv2d
            Torch kernel object.
        """
    
        # First create a kernel object
        kernel = self.object.as_kernel(pixsize) 

        torch_kernel = torch.nn.Conv2d(
            in_channels  = 1,
            out_channels = 1,
            kernel_size  = (kernel.array.shape[0], kernel.array.shape[1]),
            padding      = 'same',
            dtype        = torch.float64
        )

        torch_kernel.weight.data[0,0] = torch.from_numpy(kernel.array)
        torch_kernel.bias  .data      = torch.zeros_like(torch_kernel.bias)

        torch_kernel.weight.requires_grad_(False)
        torch_kernel.bias  .requires_grad_(False)

        return torch_kernel


    def apply(self, image):
        """
        Apply beam kernel to image.

        Parameters
        ----------
        image : np.ndarray
            Image to apply the beam to.

        Returns
        -------
        convolved_image : np.ndarray
            Input image convolved with the beam.
        """
        return convolve(image, self.kernel.array)
    
    
    def torch_apply(self, data):
        """
        Apply torch kernel.
        Assumes the last axis to contain the frequencies / channels.

        Parameters
        ----------
        data : torch.Tensor
            Data to apply the beam to.

        Returns
        -------
        dat : torch.Tensor
            Data convolved with the beam.
        """
        dat = torch.moveaxis(data, -1, 0).view(data.shape[2], 1, data.shape[0], data.shape[1])
        dat = self.torch_kernel(dat)
        dat = torch.moveaxis(dat.view(data.shape[2], data.shape[0], data.shape[1]), 0, -1)
        return dat