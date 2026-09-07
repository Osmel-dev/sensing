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
from itertools import combinations

def computeDeploymentAPUs(LRoom,M,Delta,numAPUs):
    """
    Deploy the APUs along the perimeter of the room.
    
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
    Compute the 2D antenna positions of the APUs.
    
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

def computeDevsDeployment(D,LRoom,seed):
    """
    Generate the 2D device positions.
    
    Args
    -------------
    D : number of devices
    LRoom : perimeter of the room [m]
    seed : seed for random number generation

    Returns
    -------------
    posDevs : 2D positions of the devices (D,2), [m]
    """

    sideLength = LRoom/4

    rng = np.random.default_rng(seed)
    posDevs = rng.random((D,2))*(sideLength-4) + 2

    return posDevs

def computeMeasGrid(I,sideMargin,LRoom,L):
    """
    Computes the coordinates of the points and assings a value of the radar
    cross section for each of them.
    
    Args
    -------------
    I : sq. num. of points in the grid
    sideMargin : marging between the grids and the sides of the room [m]
    LRoom : perimeter of the room [m]
    L : num. of scatterers

    Returns
    -------------
    pointsPos : 2D Cartesian coordinates of the grid points (numOfPoints,2), [m]
    radarCrossSection : complex radar cross section of the points (numOfPoints,)
    """

    numOfPointsPerDim = int(np.sqrt(I))
    
    sideLength = LRoom/4

    # generate the coordinates of the points in the grid
    pointSeparation = (sideLength - 2*sideMargin)/(numOfPointsPerDim - 1)
    coords1D = sideMargin + np.arange(numOfPointsPerDim)*pointSeparation
    xPoints, yPoints = np.meshgrid(coords1D, coords1D, indexing="xy")
    pointsPos = np.column_stack((xPoints.ravel(), yPoints.ravel()))

    # assign a non-zero radar cross section to a random subset of the points
    rng = np.random.default_rng(0)
    radarCrossSection = np.zeros(I)

    scattererIdx = rng.choice(I, size=L, replace=False)
    radarCrossSection[scattererIdx] = 1

    return pointsPos, radarCrossSection

def generateRoleAPUs(S, C):
    """
    Computes the roles of the APUs.
    
    Args
    -------------
    S : num. sensing APUs
    C : num. commun. APUs
    
    Returns
    -------------
    roleAPUsList : list of arrays indicating the roles of the APUs (numCombinations,S+C)
    """
    numAPUs = S + C

    roleAPUsList = []
    for commAPUs in combinations(range(numAPUs), C):
        role = np.zeros(numAPUs, dtype=int)
        role[list(commAPUs)] = 1
        roleAPUsList.append(role)


    return np.vstack(roleAPUsList)