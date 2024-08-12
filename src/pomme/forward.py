import torch


def image_along_last_axis(src, dtau, tau, img_bdy):
    """
    Create syntehtic image along the last data axis.

    Parameters
    ----------
    src : torch.Tensor
        Source function for the radiation transport equation.
    dtau : torch.Tensor
        Optical depth difference between two consequtive points.
    tau : torch.Tensor
        Optical depth.
    img_bdy : torch.Tensor
        Boundary condition for the image.

    Returns
    -------
    img : torch.Tensor
        Synthetic image along the last axis.
    """
    
    # Check dimensionality of the input
    assert src.dim() == tau.dim()-1, "Tensor src should only have spatial dimensions, no frequency!"

    # Get the index of the last spatial axis
    last_axis = src.dim() - 1

    src_0 = src[..., :-1]
    src_1 = src[..., +1:]

    exp_minus_tau = torch.exp(-tau)

    emt_0 = exp_minus_tau[..., :-1, :]
    emt_1 = exp_minus_tau[..., +1:, :]

    # threshold differentiating the two optical dpeth regimes
    mask_threshold = 1.0e-4
    mask = torch.Tensor(dtau < mask_threshold)

    # Case a: dtau > threshold 
    result  = torch.einsum("..., ...f -> ...f", src_0, emt_1 - emt_0 * (1.0 - dtau))
    result += torch.einsum("..., ...f -> ...f", src_1, emt_0 - emt_1 * (1.0 + dtau))
    result /= (dtau + 1.0e-30)
    # note that the regularizer 1.0e-30 is never going to be significant
    # however it is essential to avoid nans in backprop (see https://github.com/Magritte-code/pomme/issues/2)
    
    # Use a Taylor expansion for small dtau
    cc     = (1.0/2.0) * dtau
    fac_0  = cc.clone() 
    fac_1  = cc.clone()
    cc    *= (1.0/3.0) * dtau 
    fac_0 += cc
    fac_1 -= cc
    cc    *= (1.0/4.0) * dtau 
    fac_0 += cc
    fac_1 -= cc

    result[mask] = (  torch.einsum("..., ...f -> ...f", src_0, emt_0 * fac_0) \
                    + torch.einsum("..., ...f -> ...f", src_1, emt_1 * fac_1) )[mask]

    img = img_bdy * exp_minus_tau[..., -1, :] + torch.sum(result, dim=last_axis)

    return img