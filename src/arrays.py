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

def computeSteeringVecs(positionPoints,freq,posSensingAPU,posCommAPU,LRoom,antsPerAPU,interAntSpacing):
    """
        Computes the steering vectors of the sensing and communication APUs.
    
        Args
        -------------
        positionPoints : 2D coordinates of points (I**2,2), [m]
        freq : subcarriers frequency (D,1), [Hz]
        posSensingAPU : 2D coordinates of APUs (S,2), [m]
        posCommAPU : 2D coordinates of APUs (C,2), [m]
        LRoom : perimeter of the room [m]
        antsPerAPU : number of antennas per APU
        interAntSpacing : inter-antenna spacing [m]
    
        Returns
        -------------
        steeringVecComm : steering vectors of the communication APUs (M,I**2)
        steeringVecSen : steering vectors of the sensing APUs (M,I**2)
    """
    
    numPoints = positionPoints.shape[0]
    m = np.arange(antsPerAPU)
    subCarrWavelength = constants.c/freq

    # side length of the square perimeter
    sideLength = LRoom/4

    x, y = posSensingAPU

    # Determine side orientation
    if np.isclose(y, 0):                # bottom side
        ulaOrientationVectorSensing = np.array([1, 0]).T
    elif np.isclose(x, sideLength):     # right side
        ulaOrientationVectorSensing = np.array([0, 1]).T
    elif np.isclose(y, sideLength):     # top side
        ulaOrientationVectorSensing = np.array([1, 0]).T
    elif np.isclose(x, 0):              # left side
        ulaOrientationVectorSensing = np.array([0, 1]).T
    else:
        raise ValueError("Sensing APU not on any side")
    
    x, y = posCommAPU

    # Determine side orientation
    if np.isclose(y, 0):                # bottom side
        ulaOrientationVectorComm = np.array([1, 0]).T
    elif np.isclose(x, sideLength):     # right side
        ulaOrientationVectorComm = np.array([0, 1]).T
    elif np.isclose(y, sideLength):     # top side
        ulaOrientationVectorComm = np.array([1, 0]).T
    elif np.isclose(x, 0):              # left side
        ulaOrientationVectorComm = np.array([0, 1]).T
    else:
        raise ValueError("Communication APU not on any side")
    
    steeringVecSen = np.zeros((antsPerAPU,numPoints),dtype=complex)
    steeringVecComm = np.zeros((antsPerAPU,numPoints),dtype=complex)
    for pointIdx, point in enumerate(positionPoints):
        # direction of the device with respect to the sensing ULA
        displacementVector = (point - posSensingAPU)/np.linalg.norm(point - posSensingAPU)

        # sensing steering vectors
        steeringVecSen[:,pointIdx] = (np.exp(-1j*2*np.pi*(interAntSpacing/subCarrWavelength)
                                                     *m*(ulaOrientationVectorSensing @ displacementVector)))
        
        # direction of the device with respect to the comm. ULA
        displacementVector = (point - posCommAPU)/np.linalg.norm(point - posCommAPU)

        # comm steering vectors
        steeringVecComm[:,pointIdx] = (np.exp(-1j*2*np.pi*(interAntSpacing/subCarrWavelength)
                                                     *m*(ulaOrientationVectorComm @ displacementVector)))

    return steeringVecComm, steeringVecSen

def computePrecoders(posAPUs,posDevs,M,LRoom,Delta,freqPerDev,powPerAPUPerDev,idxCommAPUs):
    """
    Compute the digital joint coherent precoders. Each device is allocated to a different
    subcarrier and its precoder is computed as maximum ratio transmission. 

    Args
    -------------
    posAPUs : 2D positions of the APUs (S+C,2), [m]
    posDevs : 2D positions of the devices (D,2), [m]
    idxAPU2Dev : devices-to-APU association vector (D,1)
    M : number of antennas per APU 
    LRoom : perimeter of the room [m]
    Delta : inter-antenna spacing [m]
    freqPerDev : subcarriers frequency (D,1), [Hz]
    roleAPUs : roles of the APUs "1" communication and "0" sensing (S+C,1)

    Returns
    -------------
    precoders : digital precoders(M,C,D)
    """

    sideLength = LRoom / 4
    m = np.arange(M)

    # num. devices
    D = freqPerDev.size

    # num. commun. APUs
    C = idxCommAPUs.size

    steeringVectors = np.zeros((M,C,D), dtype=complex)
    precoders = np.zeros((M,C,D), dtype=complex)

    for dev in np.arange(D):
        for idx, cAPU in enumerate(idxCommAPUs):
            # position of c-th communication APUs
            posCommAPU = posAPUs[cAPU]
            x, y = posAPUs[cAPU]

            # determine side orientation
            if np.isclose(y, 0):                # bottom side
                ulaOrientationVector = np.array([1, 0]).T
            elif np.isclose(x, sideLength):     # right side
                ulaOrientationVector = np.array([0, 1]).T
            elif np.isclose(y, sideLength):     # top side
                ulaOrientationVector = np.array([1, 0]).T
            elif np.isclose(x, 0):              # left side
                ulaOrientationVector = np.array([0, 1]).T
            else:
                raise ValueError("APU not on any side")
            
            # direction of the device with respect to the ULA
            displacementVector = (posDevs[dev] - posCommAPU)/np.linalg.norm(posDevs[dev] - posCommAPU)

            # steering vectors
            subCarrWavelength = constants.c/freqPerDev[dev]
            steeringVectors = (np.exp(-1j*2*np.pi*(Delta/subCarrWavelength)*m*
                                             (ulaOrientationVector @ displacementVector)))
            
            # precoders (power allocation according to max-min user rates)
            precoders[:,idx,dev] = np.sqrt(powPerAPUPerDev)*steeringVectors/np.linalg.norm(steeringVectors)
            
    return precoders