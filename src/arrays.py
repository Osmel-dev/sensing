import numpy as np
from scipy import constants

def computeSteeringVecs(positionPoints,freq,posSensingAPU,posCommAPU,LRoom,antsPerAPU,interAntSpacing):
    """
    Docstring for computeSteeringVecs
    
    :param somethng: Description
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
    
    steeringVectorsSensing = np.zeros((antsPerAPU,numPoints),dtype=complex)
    steeringVectorsComm = np.zeros((antsPerAPU,numPoints),dtype=complex)
    katriRaoProduct = np.zeros((antsPerAPU**2,numPoints),dtype=complex)
    for pointIdx, point in enumerate(positionPoints):
        # direction of the device with respect to the sensing ULA
        displacementVector = (point - posSensingAPU)/np.linalg.norm(point - posSensingAPU)

        # sensing steering vectors
        steeringVectorsSensing[:,pointIdx] = (np.exp(-1j*2*np.pi*(interAntSpacing/subCarrWavelength)
                                                     *m*(ulaOrientationVectorSensing @ displacementVector)))
        
        # direction of the device with respect to the comm. ULA
        displacementVector = (point - posCommAPU)/np.linalg.norm(point - posCommAPU)

        # comm steering vectors
        steeringVectorsComm[:,pointIdx] = (np.exp(-1j*2*np.pi*(interAntSpacing/subCarrWavelength)
                                                     *m*(ulaOrientationVectorComm @ displacementVector)))
        
        # compute the Katri-Rhao product between comm. and sensing steering
        # vectors
        katriRaoProduct[:,pointIdx] = np.kron(steeringVectorsComm[:,pointIdx],steeringVectorsSensing[:,pointIdx])

    return steeringVectorsComm, steeringVectorsSensing, katriRaoProduct

def computePrecoders(posAPUs,posDevs,idxAPUDevs,M,LRoom,Delta,subCarrWavelength,roleAPUs):
    """
    Compute the digital precoders. Each device is allocated to a different
    subcarrier and its precoder is computed as MRT. 

    Args
    -------------
    posAPUs : center position APUs (numAPUs, 2)
    posDevs : 2D positions of the devices (numCommAPUs*numDevsPerAPU, 2)
    idxAPUDevs : devices to APU association vector (numCommAPUs*numDevsPerAPU, 1)
    M : number of antennas per APU 
    LRoom : perimeter of the room [m]
    Delta : inter-antenna spacing [m]
    subCarrWavelength : [m]
    roleAPUs : roles of the APUs "1" communication and "0" sensing (1, numAPUs)

    Returns
    -------------
    steeringVectors : (antsPerAPU, numDevsPerAPU*numCommAPUs)
    precoders : normalized steering vectors (antsPerAPU, numDevsPerAPU*numCommAPUs)
    """

    sideLength = LRoom // 4
    m = np.arange(M)

    numCommAPUs = roleAPUs.sum()
    numDevsPerAPU = idxAPUDevs.size // numCommAPUs

    steeringVectors = np.zeros((M,numDevsPerAPU*numCommAPUs), dtype=complex)
    precoders = np.zeros((M,numDevsPerAPU*numCommAPUs), dtype=complex)

    for dev, apu in enumerate(idxAPUDevs):
        #  print(f"dev {dev}, apu {apu}")
         
         x, y = posAPUs[apu]

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
            raise ValueError("APU not on any side")
         
         # direction of the device with respect to the ULA
         displacementVector = (posDevs[dev] - posAPUs[apu])/np.linalg.norm(posDevs[dev] - posAPUs[apu])

         # steering vectors
         steeringVectors[:,dev] = (np.exp(-1j*2*np.pi*(Delta/subCarrWavelength[dev])*m*
                                          (ulaOrientationVector @ displacementVector)))
         
         # precoders (normalized steering vectors)
         precoders[:,dev] = steeringVectors[:,dev]/np.linalg.norm(steeringVectors[:,dev])

    return steeringVectors, precoders