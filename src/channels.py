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

def pointsChannelCoeffs(posPoints,freqPerDev,posAPUs,LRoom,Delta,M,roleAPUs,radarCrossSection):
    """
    Computes the bistatic channel coefficients for from the comm. APUs to the grid points
    and back to the sensing APUs.

    Args
    -------------
    posPoints : 2D Cartesian coordinates of the grid points (I**2,2), [m]
    freqPerDev : subcarriers frequency (D,1), [Hz]
    posAPUs : 2D position APUs (S+C,2), [m]
    LRoom : perimeter of the room [m]
    Delta : inter-antenna spacing [m]
    M : number of antennas per APU 
    roleAPUs : roles of the APUs "1" communication and "0" sensing (S+C,1)
    radarCrossSection : complex radar cross section of the points (I**2,)

    Returns
    -------------
    H : complex channel coefficients (S,D,M,M)
    tau = : (S,D,I**2)
    """

    sideLength = LRoom/4

    # num. devices
    D = freqPerDev.size

    # indeces & num. of commun. APUs
    idxCommAPUs = np.flatnonzero(roleAPUs == 1)
    C = idxCommAPUs.size

    # indeces & num. of sensing APUs
    idxSenAPUs = np.flatnonzero(roleAPUs == 0)
    S = idxSenAPUs.size    

    # memory preallocation
    H = np.zeros((S,C,D,M,M),dtype=complex)
    tau = np.zeros((S,C,posPoints.shape[0]),dtype=float)
    for pointIdx, point in enumerate(posPoints):
        # skip the iteration if the radar cross section of the point is 0
        if radarCrossSection[pointIdx] == 0:
            continue

        for idxSen, sAPU in enumerate(idxSenAPUs):
            # position of the s-th sensing APU 
            posSenAPU = posAPUs[sAPU]
            x, y = posSenAPU

            # determine side orientation
            if np.isclose(y, 0):                # bottom side
                ulaOrientationVectorSen = np.array([1, 0]).T
            elif np.isclose(x, sideLength):     # right side
                ulaOrientationVectorSen = np.array([0, 1]).T
            elif np.isclose(y, sideLength):     # top side
                ulaOrientationVectorSen = np.array([1, 0]).T
            elif np.isclose(x, 0):              # left side
                ulaOrientationVectorSen = np.array([0, 1]).T
            else:
                raise ValueError("Sensing APU not on any side")
            
            # direction of the device with respect to the ULA
            displacementVectorSen = (point - posSenAPU)/np.linalg.norm(point - posSenAPU)

            # delay point to sensing APU
            delaySenAPU = np.linalg.norm(point - posSenAPU)/constants.c

            phaseSen = np.arange(M)*(ulaOrientationVectorSen @ displacementVectorSen)
            
            # iterate over the comm. APUs
            for idxComm, cAPU in enumerate(idxCommAPUs):
                # position of the c-th communication APU
                posCommAPU = posAPUs[cAPU]
                x, y = posCommAPU

                # determine side orientation
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
                
                # direction of the device with respect to the ULA
                displacementVectorComm = (point - posCommAPU)/np.linalg.norm(point - posCommAPU)

                # delay communication APU to the point
                delayCommAPU = np.linalg.norm(point - posCommAPU)/constants.c

                # round trip delay
                totalDelay = delayCommAPU + delaySenAPU

                # round trip delay matrix
                tau[idxSen,idxComm,pointIdx] = totalDelay

                phaseComm = np.arange(M)*(ulaOrientationVectorComm @ displacementVectorComm)

                for idxSubCarr, subCarrier in enumerate(freqPerDev):
                    # wavelenght of the subcarrier
                    subCarrWavelength = constants.c/subCarrier

                    # steering vector sensing
                    steeringVectorSen = (np.exp(-1j*2*np.pi*(Delta/subCarrWavelength)*phaseSen))

                    # steering vector communication 
                    steeringVectorComm = (np.exp(-1j*2*np.pi*(Delta/subCarrWavelength)*phaseComm))
                
                    # matrix of channel coefficients
                    H[idxSen,idxComm,idxSubCarr,:,:] += (radarCrossSection[pointIdx]*np.exp(-1j*2*np.pi*subCarrier*totalDelay)*
                                                np.outer(steeringVectorSen, steeringVectorComm.conj()))
                                        
    return H, tau

def devsChannelCoeffs(posDevs,freqPerDev,posAPUs,LRoom,Delta,M,idxCommAPUs):
    """
    Computes devices complex channel coefficients.

    Args
    -------------
    posDevs : 2D coordinates of devices (D,2), [m]
    freqPerDev : subcarriers frequency (D,1), [Hz]
    posAPUs : 2D coordinates of APUs (S+C,2), [m]
    LRoom : perimeter of the room [m]
    Delta : inter-antenna spacing [m]
    M : number of antennas per APU
    idxCommAPUs : indices of communication APUs

    Returns
    -------------
    H : complex channel coefficients (D,C,M)
    """

    sideLength = LRoom/4

    # num. devices
    D = posDevs.shape[0]

    # num. commun. APUs
    C = idxCommAPUs.size

    # memory preallocation
    H = np.zeros((D, C, M), dtype=complex)
    for dev in np.arange(D):
        subCarrWavelength = constants.c/freqPerDev[dev]

        for idx, cAPU in enumerate(idxCommAPUs):
            # position of c-th communication APU
            posCommAPU = posAPUs[cAPU]
            x, y = posCommAPU

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
                raise ValueError("Communication APU not on any side")
            
            # direction of the device with respect to the ULA
            displacementVector = (posDevs[dev] - posCommAPU)/np.linalg.norm(posDevs[dev] - posCommAPU)

            # matrix of channel coefficients
            H[dev,idx,:] = (np.exp(-1j*2*np.pi*(Delta/subCarrWavelength)*np.arange(M)*
                                   (ulaOrientationVector @ displacementVector)))

    return H