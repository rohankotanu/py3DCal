import numpy as np
from scipy.fftpack import dst
from scipy.fftpack import idst

def fast_poisson(Gx, Gy):
    """
    Fast Poisson solver for 2D images.
    Args:
        Gx (np.ndarray): 2D array of x-derivatives
        Gy (np.ndarray): 2D array of y-derivatives
    Returns:
        depthmap (np.ndarray): 2D array of the solution to the Poisson equation
    """
    
    height, width = Gx.shape
    
    # Compute the difference of the Gx array in the x-direction to approximate the second derivative in the x-direction (only for interior)
    Gxx = Gx[1:-1,1:-1] - Gx[1:-1,:-2]
    # Compute the difference of the Gy array in the y-direction to approximate the second derivative in the y-direction (only for interior)
    Gyy = Gy[1:-1,1:-1] - Gy[:-2,1:-1]
    
    # Combine the two second derivatives to form the source term for the Poisson equation, g
    g = Gxx + Gyy 
    
    # Apply the Discrete Sine Transform (DST) to the 2D array g (row-wise transform)
    g_sinx = dst(g, norm='ortho')

    # Apply the DST again (column-wise on the transposed array) to complete the 2D DST
    g_sinxy = dst(g_sinx.T, norm='ortho').T
    
    # Create a mesh grid of indices corresponding to the interior points (excluding the boundaries)
    x_mesh, y_mesh = np.meshgrid(range(1, width-1), range(1, height-1)) 

    # Construct the denominator for the Poisson solution based on the 2D frequency space
    denom = (2*np.cos(np.pi*x_mesh/(width-1))-2) + (2*np.cos(np.pi*y_mesh/(height-1))-2)
    
    # Divide the 2D DST coefficients by the frequency-dependent denominator to solve the Poisson equation in the frequency domain
    out = g_sinxy / denom

    # Apply the inverse DST (IDST) to the result in the x-direction
    g_x = idst(out,norm='ortho')

    # Apply the inverse DST again in the y-direction to obtain the solution in the spatial domain
    g_xy = idst(g_x.T,norm='ortho').T

    # Note: The norm='ortho' option in the DST and IDST ensures that the transforms are orthonormal, maintaining energy conservation in the transforms

    # Pad the result (which is only for the interior) with 0's at the border because we are assuming fixed boundary conditions
    depthmap = np.pad(g_xy, pad_width=1, mode='constant')

    return depthmap