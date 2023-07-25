import torch


def image_along_last_axis(src, dtau, tau):
    
    # Get the index of the last spatial axis
    last_axis = src.dim() - 2

    src_0 = src[..., :-1, :]
    src_1 = src[..., +1:, :]

    exp_minus_tau = torch.exp(-tau)

    emt_0 = exp_minus_tau[..., :-1, :]
    emt_1 = exp_minus_tau[..., +1:, :]

    # threshold differentiating the two optical dpeth regimes
    dtau_threshold = 1.0e-2

    mask_a = torch.Tensor(dtau >  dtau_threshold)
    mask_b = torch.Tensor(dtau <= dtau_threshold)

    src_0a = src_0[mask_a]
    src_0b = src_0[mask_b]
    src_1a = src_1[mask_a]
    src_1b = src_1[mask_b]

    emt_0a = emt_0[mask_a]
    emt_0b = emt_0[mask_b]
    emt_1a = emt_1[mask_a]
    emt_1b = emt_1[mask_b]

    dtau_a = dtau[mask_a]
    dtau_b = dtau[mask_b]

    # Case a: dtau > threshold    
    term_0a = src_0a * (emt_1a - emt_0a * (1.0 - dtau_a))
    term_1a = src_1a * (emt_0a - emt_1a * (1.0 + dtau_a))

    # Case b: dtau <= threshold
    cc     = (1.0/2.0) * torch.ones_like(dtau_b)
    fac_0  = cc.clone() 
    fac_1  = cc.clone()
    cc    *= (1.0/3.0) * dtau_b 
    fac_0 += cc
    fac_1 -= cc
    cc    *= (1.0/4.0) * dtau_b 
    fac_0 += cc
    fac_1 -= cc
    cc    *= (1.0/5.0) * dtau_b
    fac_0 += cc
    fac_1 -= cc

    term_0b = src_0b * emt_0b * fac_0
    term_1b = src_1b * emt_1b * fac_1

    result = torch.empty_like(dtau)
    result[mask_a] = (term_0a + term_1a) / dtau_a**2
    result[mask_b] = (term_0b + term_1b)
    result *= dtau

    img = torch.sum(result, dim=last_axis)

    return img