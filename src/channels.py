import numpy as np
from scipy import constants

def pointsChannelCoeffs(posPoints,freqPerDev,idxAPU2Dev,totalDevsPerAPU,posAPUs,LRoom,Delta,M,roleAPUs,radarCrossSection):
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
    sideLength = LRoom/4
    m = np.arange(M)
    numAPUs = posAPUs.shape[0]

    numSensingAPUs = numAPUs - roleAPUs.sum() 
    numSubCarriers = freqPerDev.size

    idxCommAPUs = np.flatnonzero(roleAPUs == 1)

    channMx = np.zeros((numSensingAPUs,numSubCarriers,M,M),dtype=complex)
    roundTripDelay = np.zeros((numSensingAPUs,numSubCarriers,posPoints.shape[0]),dtype=float)
    for pointIdx, point in enumerate(posPoints):
        # skip the iteration if the radar cross section of the point is 0
        if radarCrossSection[pointIdx] == 0:
            continue

        for dev, subCarrier in enumerate(freqPerDev):
            # index of Comm. APU operating at the subCarrier freq.            
            posCommAPU = posAPUs[idxAPU2Dev[dev]]

            # determine the orientation of the comm. APU
            x, y = posCommAPU

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
            displacementVector = (point - posCommAPU)/np.linalg.norm(point - posCommAPU)

            # delay communication APU to the point
            delayCommAPU = np.linalg.norm(point - posCommAPU)/constants.c

            subCarrWavelength = constants.c/subCarrier

            # path loss
            pathLossComm = 1 # (subCarrWavelength/(4*np.pi*np.linalg.norm(point - posCommAPU)))**2

            # steering vector
            steeringVectorComm = (np.exp(-1j*2*np.pi*(Delta/subCarrWavelength)*m*
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
                    steeringVectorSensing = (np.exp(-1j*2*np.pi*(Delta/subCarrWavelength)*m*
                                                (ulaOrientationVector @ displacementVector)))

                    # round trip delay
                    totalDelay = delayCommAPU + delaySensingAPU

                    # path loss
                    pathLossSens = 1 # (subCarrWavelength/(4*np.pi*np.linalg.norm(point - positionSensingAPU)))**2

                    # compute the index of the APU
                    idxSensingAPU = np.sum(roleAPUs[:idxAPU] == 0)

                    # matrix of channel coefficients
                    channMx[idxSensingAPU,dev,:,:] = (np.sqrt(pathLossComm*pathLossSens)*channMx[idxSensingAPU,dev,:,:] + 
                                                      radarCrossSection[pointIdx]*np.exp(-1j*2*np.pi*subCarrier*totalDelay)*
                                                      np.outer(steeringVectorSensing, steeringVectorComm.conj()))
                    
                    # round trip delay matrix
                    roundTripDelay[idxSensingAPU,dev,pointIdx] = totalDelay

    return channMx, roundTripDelay

def devsChannelCoeffs(posDevs,freqPerDev,posAPUs,LRoom,Delta,M,idxAPU2Dev):
    """
    Computes the devices channel coefficients 

    Args
    -------------
    posPoints : 2D Cartesian coordinates of the grid points (I**2 x 2)
    

    Returns
    -------------
    channMx : (S, K, M, M)
    """
    sideLength = LRoom/4
    m = np.arange(M)
    Uc = posDevs.shape[0]

    channMx = np.zeros((Uc,M),dtype=complex)
    for dev, apu in enumerate(idxAPU2Dev):
        # index of Comm. APU operating at the subCarrier freq.
        posCommAPU = posAPUs[apu]

        subCarrWavelength = constants.c/freqPerDev[dev]

        # determine the orientation of the comm. APU
        x, y = posCommAPU

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
        displacementVector = (posDevs[dev] - posCommAPU)/np.linalg.norm(posDevs[dev] - posCommAPU)

        # path loss
        pathLoss = (subCarrWavelength/(4*np.pi*np.linalg.norm(posDevs[dev] - posCommAPU)))**2
        
        # matrix of channel coefficients
        channMx[dev,:] = np.sqrt(pathLoss)*(np.exp(-1j*2*np.pi*(Delta/subCarrWavelength)*m*
                                        (ulaOrientationVector @ displacementVector)))
    return channMx