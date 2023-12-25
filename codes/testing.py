import torch
from torch_geometric.loader import DataLoader
import pickle

from sklearn import metrics

import warnings
from pandas.errors import SettingWithCopyWarning
from os import listdir
from os.path import isfile, join

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from utils.preprocess_utils import *
from utils.training_utils import *

import argparse 

parser = argparse.ArgumentParser(description='testing parser')
parser.add_argument('--use_testloader')
parser.add_argument('--new_scaler') 
parser.add_argument('--gpu_num', type = int)  
parser.add_argument('--pred_min', type = int, default = 90)

args = parser.parse_args()

GPU_NUM = args.gpu_num 
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) 

model_dir = f'final_{args.pred_min}.pkl'

if __name__ == "__main__":
        if args.use_testloader == 'yes':
            loader = torch.load(f'final_{args.pred_min}_testloader_home.pth')
            loader2 = torch.load(f'final_{args.pred_min}_testloader_away.pth')

            y_trues, y_preds, y_pred_probs = [], [], []

            model = torch.load(model_dir)
            model.eval()

            for data, data2 in zip(loader, loader2):

                out = model(data.x_norm, data2.x_norm, data.edge_index, data2.edge_index, data.batch, data.y, data.x_norm2, data2.x_norm2, data["edge_w_norm"], data2["edge_w_norm"]) 
                y_pred_prob = out.cpu().detach().numpy()
                y_pred = out.argmax(dim=1).cpu().detach().numpy()
                y_true = data.y[:,1].long().cpu().detach().numpy()
                y_preds.append(y_pred)
                y_trues.append(y_true)
                y_pred_probs.append(y_pred_prob)

            merged_array1, merged_array2 = [], []
            for arr in y_preds:
                merged_array1.extend(arr)

            for arr in y_trues:
                merged_array2.extend(arr)

            print(f"\nAccuracy : {metrics.accuracy_score(merged_array2, merged_array1)}")
            print(f"\nF1-Score for Home Win : {metrics.f1_score(merged_array1, merged_array2, average = None)[0]}")
            print(f"\nF1-Score for Away Win : {metrics.f1_score(merged_array1, merged_array2, average = None)[1]}")
            print(f"\nF1-Score for Draw : {metrics.f1_score(merged_array1, merged_array2, average = None)[2]}")

        else:
            file_dir = input("Provide your directory with test html files : ")  
            file_dir = file_dir + '/'
            files = [f for f in listdir(file_dir) if isfile(join(file_dir, f))]

            preprocess_home = preprocess_home_data(file_dir, files, args.pred_min)
            preprocess_away = preprocess_away_data(file_dir, files, args.pred_min)

            df_x3_home, results_home, match_home, xs_home, xs2_home, edge_indices_home, edge_attributes_home, flag_home, x2_home = preprocess_home.preprocess()
            df_x3_away, results_away, match_away, xs_away, xs2_away, edge_indices_away, edge_attributes_away, flag_away, x2_away = preprocess_away.preprocess()

            if args.new_scaler == 'yes':
                gs = get_scaler(flag_home, flag_away, xs_home, xs_away, edge_attributes_home, edge_attributes_away, df_x3_home, df_x3_away, x2_home, x2_away, METHOD = 'min-max')
                scalers, df_x, df_x2, df_x3, df_x3_home, df_x3_away, label_encoder, delete, cols_to_normalize1, cols_to_normalize2, cols_to_encode = gs.make_scaler()
                home_data = get_home_data(df_x, df_x2, df_x3, xs_home, df_x3_home, edge_attributes_home, edge_indices_home, label_encoder, delete, results_home, cols_to_normalize1, cols_to_normalize2, cols_to_encode, scalers, device, METHOD = 'min-max')
                away_data = get_away_data(df_x, df_x2, df_x3, xs_away, df_x3_away, edge_attributes_away, edge_indices_away, label_encoder, delete, results_away, cols_to_normalize1, cols_to_normalize2, cols_to_encode, scalers, device, METHOD = 'min-max')
                dataset_home = home_data.get_dataset()
                dataset_away = away_data.get_dataset()

                if args.pred_min == 45:
                    bs = 16
                else:
                    bs = 64

                loader = DataLoader(dataset_home, batch_size = bs)
                loader2 = DataLoader(dataset_away, batch_size = bs)

                y_trues, y_preds, y_pred_probs = [], [], []

                model = torch.load(model_dir)
                model.eval()

                for data, data2 in zip(loader, loader2):

                    out = model(data.x_norm, data2.x_norm, data.edge_index, data2.edge_index, data.batch, data.y, data.x_norm2, data2.x_norm2, data["edge_w_norm"], data2["edge_w_norm"]) 
                    y_pred_prob = out.cpu().detach().numpy()
                    y_pred = out.argmax(dim=1).cpu().detach().numpy()
                    y_true = data.y[:,1].long().cpu().detach().numpy()
                    y_preds.append(y_pred)
                    y_trues.append(y_true)
                    y_pred_probs.append(y_pred_prob)

                merged_array1, merged_array2 = [], []
                for arr in y_preds:
                    merged_array1.extend(arr)

                for arr in y_trues:
                    merged_array2.extend(arr)

                print(f"\nAccuracy : {metrics.accuracy_score(merged_array2, merged_array1)}")
                print(f"\nF1-Score for Home Win : {metrics.f1_score(merged_array1, merged_array2, average = None)[0]}")
                print(f"\nF1-Score for Away Win : {metrics.f1_score(merged_array1, merged_array2, average = None)[1]}")
                print(f"\nF1-Score for Draw : {metrics.f1_score(merged_array1, merged_array2, average = None)[2]}")

            else:
                scaler_dir = 'scalers.json'
                with open(scaler_dir, 'rb') as f:
                    scalers = pickle.load(f)
                gs = get_scaler(flag_home, flag_away, xs_home, xs_away, edge_attributes_home, edge_attributes_away, df_x3_home, df_x3_away, x2_home, x2_away, METHOD = 'min-max')
                _, df_x, df_x2, df_x3, df_x3_home, df_x3_away, label_encoder, delete, cols_to_normalize1, cols_to_normalize2, cols_to_encode = gs.make_scaler()
                home_data = get_home_data(df_x, df_x2, df_x3, xs_home, df_x3_home, edge_attributes_home, edge_indices_home, label_encoder, delete, results_home, cols_to_normalize1, cols_to_normalize2, cols_to_encode, scalers, device, METHOD = 'min-max')
                away_data = get_away_data(df_x, df_x2, df_x3, xs_away, df_x3_away, edge_attributes_away, edge_indices_away, label_encoder, delete, results_away, cols_to_normalize1, cols_to_normalize2, cols_to_encode, scalers, device, METHOD = 'min-max')
                dataset_home = home_data.get_dataset()
                dataset_away = away_data.get_dataset()

                if args.pred_min == 45:
                    bs = 16
                else:
                    bs = 64

                loader = DataLoader(dataset_home, batch_size = bs)
                loader2 = DataLoader(dataset_away, batch_size = bs)

                model = torch.load(model_dir)
                model.eval()

                for data, data2 in zip(loader, loader2):

                    out = model(data.x_norm, data2.x_norm, data.edge_index, data2.edge_index, data.batch, data.y, data.x_norm2, data2.x_norm2, data["edge_w_norm"], data2["edge_w_norm"]) 
                    y_pred_prob = out.cpu().detach().numpy()
                    y_pred = out.argmax(dim=1).cpu().detach().numpy()
                    y_true = data.y[:,1].long().cpu().detach().numpy()
                
                if y_true == 0:
                    true_label = 'Home Win'
                elif y_true == 1:
                    true_label = 'Away Win'
                else:
                    true_label = 'Draw'

                if y_pred == 0:
                    pred_label = 'Home Win'
                elif y_pred == 1:
                    pred_label = 'Away Win'
                else:
                    pred_label = 'Draw'

                print(f"True Label : {true_label}")
                print(f"\nPredicted Label : {pred_label}")

            
            
