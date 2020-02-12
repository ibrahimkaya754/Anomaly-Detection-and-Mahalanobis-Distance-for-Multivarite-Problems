import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from kerasbackend import kerasbackend
KERASBACKEND = kerasbackend('tensorflow')

from keras_contrib.optimizers import Yogi
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.models import Range1d
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
output_notebook()

TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select"
from keras.layers import Input, Dense, Dropout, regularizers
from keras.models import Model, load_model
from keras.layers.normalization import BatchNormalization
from livelossplot import PlotLossesKeras
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam, SGD
from keras import backend as k
from keras.utils import plot_model
from keras.utils import multi_gpu_utils
import tensorflow as tf

from keras.layers import Conv1D
from keras.layers import MaxPooling1D, Flatten

from keras.utils import plot_model

# Pandas Libraries
import pandas as pd

# Numpy Libraries
import numpy as np
np.random.seed(8)

# File IO Libraries
import glob
import scipy.io as sio
import pickle

# Plotting Libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.models import Range1d

# Data Preparation Libraries
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
# Scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from matplotlib import pyplot

import kerasbackend
from keras import backend as K

from keras_contrib.optimizers import Padam, Yogi, ftml
from keras.optimizers import SGD
from keras.layers.merge import Concatenate
from keras.layers import Multiply
from keras.layers.core import Lambda
from ipy_table import *

###################### Tensorflow Ram Kullanımının optimize edilmesi ####################################################################

if KERASBACKEND.KERAS_BACKEND == 'tensorflow':
    # TensorFlow wizardry
    config = tf.ConfigProto() 
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True 
    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 1.0 
    # Create a session with the above options specified.
    K.tensorflow_backend.set_session(tf.Session(config=config))

##################################################################################################################################
