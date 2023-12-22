import re
import pandas as pd 
import json
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
import torch

from os import listdir
from os.path import isfile, join

import warnings
from pandas.errors import SettingWithCopyWarning
import pickle

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from preprocess_utils import * 

import argparse 

parser = argparse.ArgumentParser(description='preprocessing parser')
parser.add_argument('--gpu_num', type = int)  
parser.add_argument('--pred_min', type = int, default = 90)  

args = parser.parse_args()

GPU_NUM = args.gpu_num
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU

directory = input("Provide your directory with html files : ")  # This must end with /
directory = directory + '/'
files = [f for f in listdir(directory) if isfile(join(directory, f))]

preprocess_home = preprocess_home_data(directory, files, args.pred_min)
preprocess_away = preprocess_away_data(directory, files, args.pred_min)

if __name__ == "__main__":

    df_x3_home, results_home, match_home, xs_home, xs2_home, edge_indices_home, edge_attributes_home, flag_home, x2_home = preprocess_home.preprocess()
    df_x3_away, results_away, match_away, xs_away, xs2_away, edge_indices_away, edge_attributes_away, flag_away, x2_away = preprocess_away.preprocess()

    gs = get_scaler(flag_home, flag_away, xs_home, xs_away, edge_attributes_home, edge_attributes_away, df_x3_home, df_x3_away, x2_home, x2_away, METHOD = 'min-max')

    scalers, df_x, df_x2, df_x3, df_x3_home, df_x3_away, label_encoder, delete, cols_to_normalize1, cols_to_normalize2, cols_to_encode = gs.make_scaler()

    home_data = get_home_data(df_x, df_x2, df_x3, xs_home, df_x3_home, edge_attributes_home, edge_indices_home, label_encoder, delete, results_home, cols_to_normalize1, cols_to_normalize2, cols_to_encode, scalers, device, METHOD = 'min-max')
    away_data = get_away_data(df_x, df_x2, df_x3, xs_away, df_x3_away, edge_attributes_away, edge_indices_away, label_encoder, delete, results_away, cols_to_normalize1, cols_to_normalize2, cols_to_encode, scalers, device, METHOD = 'min-max')

    dataset_home = home_data.get_dataset()
    dataset_away = away_data.get_dataset()

    with open("scalers.json", 'wb') as f:
        pickle.dump(scalers, f) 

    with open(f"final_home_{args.pred_min}.json", 'wb') as f:
        pickle.dump(dataset_home, f) 
 
    with open(f"final_away_{args.pred_min}.json", 'wb') as f:
        pickle.dump(dataset_away, f) 