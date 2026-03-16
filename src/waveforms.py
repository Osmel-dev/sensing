import numpy as np
from scipy import constants

def subCarriersGen(fc,K,subCarrierSpacing):
    """
    Generates the subcarriers around the central frequency fc. 

    Args
    -------------
    fc : central operation frequency [Hz]
    K : number of subcarriers used
    subCarrierSpacing : inter subcarrier spacing [Hz]

    Returns
    subCarriersFreq : subcarriers frequency (nSubCarriers,1) [Hz]
    -------------
    """
    if K % 2:
        subCarriersIdx = np.arange(-(K-1)//2, (K-1)//2 + 1)
        subCarriersFreq = (fc + subCarrierSpacing*subCarriersIdx)
    else:
        subCarriersIdx = np.arange(-K//2, K//2)
        subCarriersFreq = (fc + subCarrierSpacing*(subCarriersIdx+0.5))

    subCarrierWavelength = constants.c/subCarriersFreq

    return subCarriersFreq, subCarrierWavelength

def gen16QAM(N,K,seed=0):
    """
    Generates the 16QAM symbols

    Args.
    ----------
    N : number of 16QAM symbols
    K : numb. of allocated subcarriers
    seed : seed of the random symbol generator

    Returns
    -------
    normalized symbols (K, N)
    """

    rng = np.random.default_rng(seed)
    I = 2*rng.integers(0, 4, size=(K,N)) - 3 # in-phase component
    Q = 2*rng.integers(0, 4, size=(K,N)) - 3 # quadrature component

    # normalized symbol -> (nSubCarriers,nUsersPerRB,nSymbols)
    return ((I + 1j*Q)/np.sqrt(10)).astype(np.complex64) 

def genOFDMSym(precoders,powAllocation,K,qamSymbs,M,N):
    """
    Generates the OFDM symbols in the frequency domain

    Parameters
    ----------
    precoders : : normalized steering vectors (M, U)
    powAllocation : power allocation comm. APUs
    K : num. of allocated subcarriers
    qamSymbs : normalized symbols (K, N)
    M : number of antennas per APU 
    N : number of 16QAM symbols

    Returns
    -------
    ofdmSymbs : ofdm symbols (M,N,K)
    """

    ofdmSymbs = np.zeros((M,N,K),dtype=complex)
    for k in range(K):
        ofdmSymbs[:,:,k] = precoders[:,k][:,None] @ qamSymbs[k,:][None,:]

    return ofdmSymbs