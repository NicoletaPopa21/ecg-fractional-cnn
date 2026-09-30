import numpy as np
from scipy.signal import lfilter, filtfilt
from scipy.special import gamma

def apply_grunwald_letnikov(data, alpha=0.6, window=15):
    """
    Applies the Grünwald-Letnikov fractional derivative to amplify finite differences in the signal.
    
    Args:
        data (np.ndarray): The input ECG signal array.
        alpha (float): The fractional order of the derivative.
        window (int): The window size for the filter.
        
    Returns:
        np.ndarray: The filtered signal array.
    """
    w = np.zeros(window)
    w[0] = 1.0
    for m in range(1, window):
        w[m] = w[m-1] * (1 - (alpha + 1) / m)
    
    return lfilter(w, [1.0], data, axis=1).astype('float32')

def apply_caputo(data, alpha=0.6, window=15):
    """
    Applies the Caputo fractional derivative to emphasize active pathological zones continuously.
    
    Args:
        data (np.ndarray): The input ECG signal array.
        alpha (float): The fractional order of the derivative.
        window (int): The window size for the filter.
        
    Returns:
        np.ndarray: The filtered signal array.
    """
    j = np.arange(window)
    weights = ((j + 1)**(1 - alpha) - j**(1 - alpha)) / gamma(2 - alpha)
    
    # Calculate the first derivative (finite differences) as required by Caputo definition
    dx = np.diff(data, axis=1, prepend=data[:, 0:1])
    
    return filtfilt(weights, [1.0], dx, axis=1).astype('float32')
