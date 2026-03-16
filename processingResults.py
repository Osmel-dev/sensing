import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from src import deployment,waveforms,utils,core
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pickle
import os
import argparse
import time

with open(f'results/results_0.pkl', 'rb') as f:
            data = pickle.load(f)

