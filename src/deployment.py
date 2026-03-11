import numpy as np
from scipy import constants

def computeDeploymentAPUs(LRoom,M,Delta,numAPUs):
    """
    Deploy the APUs along the perimeter of the room
    
    Args
    -------------
    LRoom : perimeter of the room [m]
    M : number of antennas per APU 
    Delta : inter-antenna spacing [m]
    numAPUs : total number of APUs (comm. + sensing)

    Returns
    -------------
    interAPUSpacing : inter-APU spacing [m]
    posAPUs : 2D positions of the APUs center (numberOfAPUs, 2)
    """

    # physical length of each APU
    lengthAPU = (M - 1)*Delta

    # side length of the square perimeter
    sideLength = LRoom/4

    # number of APUs on each side of the room must be equal. 
    if numAPUs % 4 != 0:
        raise ValueError("Total number of APUs must be divisible by 4.")
    else: 
        numAPUsPerSide = numAPUs//4

    # inter-APU spacing (last-to-first antenna) must be at least TWO times the
    # inter-antenna separation
    interAPUSpacing = sideLength/numAPUsPerSide - lengthAPU
    
    if interAPUSpacing < Delta*2:
        raise ValueError(
            f"Inter-APU space violation: {interAPUSpacing: .3f} m"
            f"Delta minimum required {Delta*2: .3f} m"
            )

    # Deployment of the APUs which places the first APU of each side half
    # inter-APU separation from the startig point to prevent APUs from being
    # placed on the corners.
    startAPURelative = interAPUSpacing/2 + np.arange(numAPUsPerSide) * (lengthAPU + interAPUSpacing)
    centersAPURelative = startAPURelative + lengthAPU/2.0
    idxAPU = 0

    posAPUs = np.zeros((numAPUsPerSide*4,2))

    # side 1: bottom edge (y = 0), x from 0 -> sideLength
    for x in centersAPURelative:
        posAPUs[idxAPU, :] = [x, 0.0]
        idxAPU += 1

    # side 2: right edge (x = sideLength), y from 0 -> sideLength
    for y in centersAPURelative:
        posAPUs[idxAPU, :] = [sideLength, y]
        idxAPU += 1

    # side 3: top edge (y = sideLength), x from sideLength -> 0
    for x in centersAPURelative[::-1]:
        posAPUs[idxAPU, :] = [x, sideLength]
        idxAPU += 1

    # side 4: left edge (x = 0), y from sideLength -> 0
    for y in centersAPURelative[::-1]:
        posAPUs[idxAPU, :] = [0.0, y]
        idxAPU += 1
    
    return posAPUs, interAPUSpacing

def computeAntPositions(posAPUs,antsPerAPU,Delta,LRoom):
    """
    Compute the 2D antenna positions of the APUs
    
    Args
    -------------
    posAPUs : center position APUs (numberOfAPUs, 2)
    antsPerAPU : number of antennas per APU 
    Delta : inter-antenna spacing [m]
    LRoom : perimeter of the room [m]

    Returns
    -------------
    posAnts : 2D positions of the APUs antennas (numAPUs, antsPerAPU, 2)
    """

    numAPUs = posAPUs.shape[0]
    posAnts = np.zeros((numAPUs, antsPerAPU, 2))
    
    # side lenght 
    sideLength = LRoom/4

    # Antenna offsets. Shifts the antenna positions to be centred around 0.
    offsets = (np.arange(antsPerAPU) - (antsPerAPU-1)/2) * Delta

    # antenna deployment
    for i in range(numAPUs):
        x, y = posAPUs[i]

        # Determine side orientation
        if np.isclose(y, 0):                # bottom side
            dx, dy = 1, 0
        elif np.isclose(x, sideLength):     # right side
            dx, dy = 0, 1
        elif np.isclose(y, sideLength):     # top side
            dx, dy = 1, 0
        elif np.isclose(x, 0):              # left side
            dx, dy = 0, 1
        else:
            raise ValueError("APU not on any side")

        # Compute per-element positions
        posAnts[i,:,0] = x + offsets * dx
        posAnts[i,:,1] = y + offsets * dy

    return posAnts

def computeDevsDeployment(posAPUs,U,Umax,roleAPUs,LRoom,seed):
    """
    Compute the 2D device positions corresponding to each APU
    
    Args
    -------------
    posAPUs : center position APUs (numAPUs, 2)
    U : number of devices assigned to each APU
    Umax : max. num. devs an APU can serve
    sectorRad : maximum distance from each APU to its associated devices [rmin, rmax]
    sectorAngle : opening APU angle where the served devices lie
    roleAPUs : roles of the APUs "1" communication and "0" sensing (1, numAPUs)
    LRoom : perimeter of the room [m]

    Returns
    -------------
    posDevs : 2D positions of the devices (U, 2)
    idxAPUDevs : APU index serving the u-th dev. (U, 1)
    """

    sideLength = LRoom/4

    # devs deployment [2, sideLength-2]
    rng = np.random.default_rng(seed)
    posDevs = rng.random((U,2))*(sideLength-4) + 2

    # positions,num., and indeces of the comm. APUs
    posCommAPUs = posAPUs[roleAPUs==1,:]
    numCommAPUs = roleAPUs.sum()
    idxCommAPUGlobal = np.flatnonzero(roleAPUs)

    # check if the network capacity meets the demads
    if numCommAPUs * Umax < U:
        raise ValueError("Not enough total capacity to serve all devices.")
     
    # pairwise distances APUs-to-devs via Euclidean norm (U,T)
    dist = np.linalg.norm(posDevs[:, None, :] - posCommAPUs[None, :, :], axis=2)

    # orders by increasing distance the matrix dist and returns the (u,t)
    # indexes of the ordered matrix
    pairs = np.array(
        np.unravel_index(
            np.argsort(dist, axis=None), dist.shape)).T
    
    # 
    capacity = np.full(numCommAPUs, Umax, dtype=int)
    idxAPU2Dev = np.full(U, -1, dtype=int) 

    for idxDev, idxAPULocal in pairs:
        if idxAPU2Dev[idxDev] == -1 and capacity[idxAPULocal] > 0:
            idxAPU2Dev[idxDev] = idxCommAPUGlobal[idxAPULocal]
            capacity[idxAPULocal] -= 1

        if np.all(idxAPU2Dev != -1) or np.all(capacity == 0):
            break   

    return posDevs, idxAPU2Dev

def computeMeasGrid(I,sideMargin,LRoom,L,seed=None):
    """
    Computes the coordinates of the points and assings a value of the radar
    cross section for each of them.
    
    Args
    -------------
    I : sq. num. of points in the grid
    sideMargin : marging between the grids and the sides of the room [m]
    LRoom : perimeter of the room [m]
    L : num. of scatterers
    seed : seed for selecting randomly the positions of the scatterers

    Returns
    -------------
    pointsPos : 2D Cartesian coordinates of the grid points (numOfPoints x 2)
    radarCrossSection : complex radar cross section of the points (numOfPoints,)
    """

    numOfPointsPerDim = int(np.sqrt(I))

    if numOfPointsPerDim**2 != I:
        raise ValueError("Number of points must be a perfect square.")
    
    sideLength = LRoom/4
    pointSeparation = (sideLength - 2*sideMargin)/(numOfPointsPerDim - 1)

    coords1D = sideMargin + np.arange(numOfPointsPerDim)*pointSeparation
    
    xPoints, yPoints = np.meshgrid(coords1D, coords1D, indexing="xy")
    pointsPos = np.column_stack((xPoints.ravel(), yPoints.ravel()))

    rng = np.random.default_rng(seed)
    radarCrossSection = np.zeros(I)

    scattererIdx = rng.choice(I, size=L, replace=False)
    radarCrossSection[scattererIdx] = 1

    return pointsPos, radarCrossSection