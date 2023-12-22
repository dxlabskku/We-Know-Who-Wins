import pandas as pd 
import numpy as np
from tqdm import tqdm
import torch
import time
import pickle

from torch_geometric.nn import GCNConv

import warnings
from pandas.errors import SettingWithCopyWarning

from utils.training_utils import *

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

import argparse 

parser = argparse.ArgumentParser(description='training parser')
parser.add_argument('--gpu_num', type = int)  
parser.add_argument('--n_epochs', type = int, default = 1000)
parser.add_argument('--e_patience', type = int, default = 50)    
parser.add_argument('--lr', type = float, default = 0.001)  
parser.add_argument('--pred_min', type = int, default = 90)

args = parser.parse_args()

GPU_NUM = args.gpu_num
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) 

home_directory = 'final_home_90.json'
away_directory = 'final_away_90.json'

with open(home_directory, 'rb') as f:
    dataset_home = pickle.load(f)

with open(away_directory, 'rb') as f:
    dataset_away = pickle.load(f)

if __name__ == "__main__":

    batchsize = 64
    test_size = 0.2 
    valid_size = 0.25
    random_state = 42
    mdl = make_data_loader(dataset_home, dataset_away, device)
    train_loader, train_loader2, valid_loader, valid_loader2, test_loader, test_loader2, class_weights = mdl.get_dataloader(batchsize, test_size, valid_size, random_state)

    torch.save(train_loader, f'final_{args.pred_min}_trainloader_home.pth')
    torch.save(train_loader2,f'final_{args.pred_min}_trainloader_away.pth')
    torch.save(valid_loader, f'final_{args.pred_min}_validloader_home.pth')
    torch.save(valid_loader2, f'final_{args.pred_min}_validloader_away.pth')
    torch.save(test_loader, f'final_{args.pred_min}_testloader_home.pth')
    torch.save(test_loader2, f'final_{args.pred_min}_testloader_away.pth')

    TARGET = 1
    N_EPOCHS = args.n_epochs
    E_PATIENCE = args.e_patience
    LEARNING_RATE = args.lr

    model = GNN(input_size = 7, hidden_channels = 128, hidden_channels2 = 128, hidden_channels3 = 64, mid_channel = 64, final_channel = 16, len_added = 16, num_classes = 3, conv = GCNConv)

    train_util = train_utils()

    model, results = train_util.train_test(
        device, train_loader, train_loader2, valid_loader, valid_loader2, model, class_weights, target = TARGET, edge_col_name = 'edge_w_norm',
        learning_rate=LEARNING_RATE, e_patience = E_PATIENCE, n_epochs=N_EPOCHS)

    torch.save(model.state_dict(), f'final_{args.pred_min}.json')
    torch.save(model, f'final_{args.pred_min}.pkl')
