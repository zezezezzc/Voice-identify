import os
import math
import numpy
from numpy import array

keys = ['carm', 'noisy', 'traffic', 'tv']

train_data_dir = ''
test_data_dir = ''

# classifier_dir = ''
# feature_dir = ''
# train_file = ''
# test_file = ''

NUM_OF_NOISE = len(keys)
NUM_OF_MIXTURE = len(keys)
NUM_OF_ITERATION = 1000
COVARIANCE_TYPE = 'diag'
NUM_DECISION_FRAME = 100
DIM_OF_MFCC_FEAT = 13
SAVE_MAT = 1
RESET = 0
SAMPLING_RATE = 16000
FRAME_LEN_MS = 0.008
INPUT_LEN_MS = 0.004
NFFT = 128
PRE_EMPHASIS_COEF = 0
FPRE_EMPHASIS_COEF = 0.97
USING_ROUND = 0

# FEATURE, NORMALIZATION, SHUFFLE, LOGSCALE
settings = [[1, 1, 0, 1, 0]]

k = array(range(0, NFFT//2+1))
freq_weight = numpy.exp(-1j * 2 * math.pi * k / NFFT) * (-FPRE_EMPHASIS_COEF) + 1

