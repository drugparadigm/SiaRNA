import os
import time
import random
import json
import lxml
import importlib
import subprocess

import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import pandas as pd
import numpy as np
import pickle as pkl
import plotly.graph_objects as go
import umap.umap_ as umap


from plotly.subplots import make_subplots
from tqdm import tqdm
from tqdm.notebook import tqdm
from pandarallel import pandarallel
from ast import literal_eval
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

from Bio import SeqUtils

from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.utils import shuffle



from IPython.core.display import display, HTML
pandarallel.initialize(progress_bar = True)
tqdm.pandas()
