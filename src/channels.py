import numpy as np
from scipy import constants

def computeChannelCoeffs(posPoints,subCarrierFreq,totalDevsPerAPU,posAPUs,LRoom,Delta,M,roleAPUs,radarCrossSection):
    """
    Computes the channel coefficients for from the comm. APUs to the grid points
    and back to the sensing APUs.

    Args
    -------------
    posPoints : 2D Cartesian coordinates of the grid points (I**2 x 2)
    subCarrierFreq : subcarriers frequency (K,1) [Hz]
    totalDevsPerAPU :
    posAPUs : center position APUs (S+C, 2)
    LRoom : perimeter of the room [m]
    Delta : inter-antenna spacing [m]
    M : number of antennas per APU 
    roleAPUs : roles of the APUs "1" communication and "0" sensing (S+C,1)
    radarCrossSection : complex radar cross section of the points (I**2,)

    Returns
    -------------
    channMx : (S, K, M, M)
    roundTripDelay = : (S,K,I**2)
    """
    subCarrWavelength = constants.c/subCarrierFreq

    sideLength = LRoom/4
    m = np.arange(M)
    numAPUs = posAPUs.shape[0]

    numSensingAPUs = numAPUs - roleAPUs.sum() 
    numSubCarriers = subCarrierFreq.size

    idxCommAPUs = np.flatnonzero(roleAPUs == 1)

    channMx = np.zeros((numSensingAPUs,numSubCarriers,M,M),dtype=complex)
    roundTripDelay = np.zeros((numSensingAPUs,numSubCarriers,posPoints.shape[0]),dtype=float)
    for pointIdx, point in enumerate(posPoints):
        # skip the iteration if the radar cross section of the point is 0
        if radarCrossSection[pointIdx] == 0:
            continue

        for subCarrierIdx, subCarrier in enumerate(subCarrWavelength):
            # index of Comm. APU operating at the subCarrier freq.
            positionCommAPU = posAPUs[idxCommAPUs[subCarrierIdx // totalDevsPerAPU]]

            # determine the orientation of the comm. APU
            x, y = positionCommAPU

            # Determine side orientation
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
            displacementVector = (point - positionCommAPU)/np.linalg.norm(point - positionCommAPU)

            # delay communication APU to the point
            delayCommAPU = np.linalg.norm(point - positionCommAPU)/constants.c

            # steering vector
            steeringVectorComm = (np.exp(-1j*2*np.pi*(Delta/subCarrier)*m*
                                         (ulaOrientationVector @ displacementVector)))
            
            # iterate over the sensing APUs
            for idxAPU, role in enumerate(roleAPUs):
                if role == 0: 
                    positionSensingAPU = posAPUs[idxAPU]

                    x, y = positionSensingAPU

                    # Determine side orientation
                    if np.isclose(y, 0):                # bottom side
                        ulaOrientationVector = np.array([1, 0]).T
                    elif np.isclose(x, sideLength):     # right side
                        ulaOrientationVector = np.array([0, 1]).T
                    elif np.isclose(y, sideLength):     # top side
                        ulaOrientationVector = np.array([1, 0]).T
                    elif np.isclose(x, 0):              # left side
                        ulaOrientationVector = np.array([0, 1]).T
                    else:
                        raise ValueError("Sensing APU not on any side")
                    
                    # direction of the device with respect to the ULA
                    displacementVector = (point - positionSensingAPU)/np.linalg.norm(point - positionSensingAPU)

                    # delay point to sensing APU
                    delaySensingAPU = np.linalg.norm(point - positionSensingAPU)/constants.c

                    # steering vector
                    steeringVectorSensing = (np.exp(-1j*2*np.pi*(Delta/subCarrier)*m*
                                                (ulaOrientationVector @ displacementVector)))

                    # round trip delay
                    totalDelay = delayCommAPU + delaySensingAPU

                    # compute the index of the APU
                    idxSensingAPU = np.sum(roleAPUs[:idxAPU] == 0)

                    # matrix of channel coefficients
                    channMx[idxSensingAPU,subCarrierIdx,:,:] = (channMx[idxSensingAPU,subCarrierIdx,:,:] + 
                                                            radarCrossSection[pointIdx]*np.exp(-1j*2*np.pi*subCarrierFreq[subCarrierIdx]*totalDelay)*
                                                                                               np.outer(steeringVectorSensing, steeringVectorComm.conj()))
                    
                    # round trip delay matrix
                    roundTripDelay[idxSensingAPU,subCarrierIdx,pointIdx] = totalDelay

    return channMx, roundTripDelay