"""
Functions used for the simulations presented in:

O. Martínez Rosabal and O. L. A. López,
"Radio Stripe-Based Distributed ISAC System with Dynamic Sensing-Communication
Reconfiguration," arXiv:2604.08982, 2026.

Version: 1.0
Last modified: 2026-09-07

License: MIT
See the LICENSE file in the repository root.

If you use this code in research resulting in a publication, please cite
the paper above.
"""

import numpy as np
from scipy import constants

def subCarriersGen(fc,subCarrierSpacing):
    """
    Generates the subcarriers around the central frequency fc. 

    Args
    -------------
    fc : central operation frequency [Hz]
    subCarrierSpacing : inter subcarrier spacing [Hz]

    Returns
    subCarriersFreq : subcarriers frequency (nSubCarriers,1), [Hz]
    -------------
    """
    
    subCarriersFreq = np.arange(-32,32)*subCarrierSpacing + fc

    subCarrierWavelength = constants.c/subCarriersFreq

    return subCarriersFreq, subCarrierWavelength

def gen16QAM(D,seed=0):
    """
    Generates the 16QAM symbols

    Args.
    ----------
    D : numb. devices
    seed : seed of the random symbol generator

    Returns
    -------
    normalized symbols (D,)
    """

    rng = np.random.default_rng(seed)
    I = 2*rng.integers(0, 4, size=(D,)) - 3 # in-phase component
    Q = 2*rng.integers(0, 4, size=(D,)) - 3 # quadrature component

    # normalized symbol -> (nSubCarriers,nUsersPerRB,nSymbols)
    return ((I + 1j*Q)/np.sqrt(10)).astype(np.complex64) 

def genOFDMSym(precoders,qamSymbs):
    """
    Generates the OFDM symbols in the frequency domain

    Parameters
    ----------
    precoders : digial precoders (M,C,D)
    qamSymbs : normalized symbols (D,)

    Returns
    -------
    ofdmSymbs : ofdm symbols (M*C,D)
    """
    M, C, D = precoders.shape

    distribPrecoders = np.transpose(precoders, (1,0,2)).reshape(C*M,D)
    ofdmSymbs = distribPrecoders*qamSymbs.T[None, :]

    return ofdmSymbs