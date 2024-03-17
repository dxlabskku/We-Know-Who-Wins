import pandas as pd 
import numpy as np
from tqdm import tqdm
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.loader import DataLoader
import time

from torch.nn import Linear, Softmax, ELU
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from sklearn import metrics

#import warnings
#from pandas.errors import SettingWithCopyWarning

#warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class make_data_loader:
    def __init__(self, dataset_home, dataset_away, device):
        self.dataset_home = dataset_home 
        self.dataset_away = dataset_away 
        self.device = device 

    def prepare_split(self):
        for i in range(len(self.dataset_home)):

            y_i = self.dataset_home[i].y.cpu().detach().numpy()
            y = y_i if i ==0 else np.vstack([y, self.dataset_home[i].y.cpu().detach().numpy()])
            
        df_y = pd.DataFrame(y).melt()

        df_y1 = df_y.groupby(['variable'],as_index=False).agg({'value':['sum', 'count']})
        df_y1.columns = ['variable','sum', 'count']
        df_y1['mean'] = df_y1['sum']/df_y1['count']
        df_y1['missing_values'] = (1 - df_y1['count'] / len(self.dataset_home))

        for c in ['mean', 'missing_values']:
            df_y1[c] = df_y1[c].apply(lambda x:round(x*100,2))

        TARGET = 1

        dataset_target, dataset_target2, Y = [], [], []

        for i in range(len(self.dataset_home)):
            if not(self.dataset_home[i]['y'][0,TARGET].isnan()):
                Y.append(self.dataset_home[i]['y'][0,TARGET])
                dataset_target.append(self.dataset_home[i])
                dataset_target2.append(self.dataset_away[i])
                
        Y = pd.DataFrame([y.cpu().detach().numpy() for y in Y]).reset_index().rename(columns={0:'target'})

        class_weights = torch.tensor([(Y.groupby('target').count().reset_index()['index'][0]/len(Y)).astype(np.float32), 
                              (Y.groupby('target').count().reset_index()['index'][1]/len(Y)).astype(np.float32),
                              (Y.groupby('target').count().reset_index()['index'][2]/len(Y)).astype(np.float32)])
        
        class_weights = class_weights.to(self.device)

        return dataset_target, dataset_target2, Y, class_weights


    def split_dataset(self, test_size = 0.2, valid_size = 0.25, random_state = None):
        TARGET = 1

        dataset_target, dataset_target2, Y, class_weights = self.prepare_split()

        random_state = np.random.randint(10**3) if random_state is None else random_state
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state = random_state)

        train_dataset, train_dataset2, test_dataset, test_dataset2 = [], [], [], []
        for train_index, test_index in sss.split(Y['index'], Y['target']):
            #print(train_index, test_index)
            train_dataset += [dataset_target[i] for i in list(train_index)]
            train_dataset2 += [dataset_target2[i] for i in list(train_index)]
            test_dataset += [dataset_target[i] for i in list(test_index)]
            test_dataset2 += [dataset_target2[i] for i in list(test_index)]

        Y2 = []
        for i in range(len(train_dataset)):
            if not(train_dataset[i]['y'][0,TARGET].isnan()):
                Y2.append(train_dataset[i]['y'][0,TARGET])
            
        Y2 = pd.DataFrame([y.cpu().detach().numpy() for y in Y2]).reset_index().rename(columns={0:'target'})

        sss = StratifiedShuffleSplit(n_splits=1, test_size=valid_size, random_state = random_state)
        train_dataset_real, train_dataset2_real, valid_dataset, valid_dataset2 = [], [], [], []

        for train_index, test_index in sss.split(Y2['index'], Y2['target']):
            #print(train_index, test_index)
            train_dataset_real += [train_dataset[i] for i in list(train_index)]
            train_dataset2_real += [train_dataset2[i] for i in list(train_index)]
            valid_dataset += [train_dataset[i] for i in list(test_index)]
            valid_dataset2 += [train_dataset2[i] for i in list(test_index)]

        print(f'Number of training graphs: {len(train_dataset_real)} -> {round(len(train_dataset_real)/len(dataset_target)*100)}%')
        print(f"Number of validation graphs: {len(valid_dataset)} -> {round(len(valid_dataset)/len(dataset_target)*100)}%")
        print(f'Number of test graphs: {len(test_dataset)} -> {round(len(test_dataset)/len(dataset_target)*100)}%')
        
        return train_dataset_real, train_dataset2_real, valid_dataset, valid_dataset2, test_dataset, test_dataset2, class_weights

    def get_dataloader(self, batchsize, test_size, valid_size, random_state):
        train_dataset_real, train_dataset2_real, valid_dataset, valid_dataset2, test_dataset, test_dataset2, class_weights = self.split_dataset(test_size, valid_size, random_state)

        bs = batchsize

        train_loader = DataLoader(train_dataset_real, batch_size = bs)
        train_loader2 = DataLoader(train_dataset2_real, batch_size = bs)
        valid_loader = DataLoader(valid_dataset, batch_size = bs)
        valid_loader2 = DataLoader(valid_dataset2, batch_size = bs)
        test_loader = DataLoader(test_dataset, batch_size = bs)
        test_loader2 = DataLoader(test_dataset2, batch_size = bs)

        return train_loader, train_loader2, valid_loader, valid_loader2, test_loader, test_loader2, class_weights
    

class GNN(torch.nn.Module):
    def __init__(self, input_size, hidden_channels, hidden_channels2, hidden_channels3, mid_channel, final_channel, len_added, num_classes, conv, conv_params={}):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = conv(
            input_size, hidden_channels, **conv_params)
        self.conv2 = conv(
            hidden_channels, hidden_channels2, **conv_params)
        self.conv3 = conv(
            hidden_channels2, hidden_channels3, **conv_params)
        
        self.lin = Linear((hidden_channels3+len_added), mid_channel)
        self.lin1 = Linear(mid_channel*2, final_channel)
        self.lin2 = Linear(final_channel, num_classes)

        self.elu = ELU()
        self.softmax = Softmax(dim = 1)

    def forward(self, x, x2, edge_index, edge_index2, batch, half_y, x_norm2_1, x_norm2_2, edge_col = None, edge_col2 = None):
        # 1. Obtain node embeddings 

        x = self.elu(self.conv1(x, edge_index, edge_col))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.elu(self.conv2(x, edge_index, edge_col))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.elu(self.conv3(x, edge_index, edge_col))
        x = F.dropout(x, p = 0.5, training = self.training)

        x2 = self.elu(self.conv1(x2, edge_index2, edge_col2))
        x2 = F.dropout(x2, p=0.5, training=self.training)

        x2 = self.elu(self.conv2(x2, edge_index2, edge_col2))
        x2 = F.dropout(x2, p=0.5, training=self.training)

        x2 = self.elu(self.conv3(x2, edge_index2, edge_col2))
        x2 = F.dropout(x2, p = 0.5, training = self.training)

        # 2. Readout layer
        batch1 = batch
        batch2 = batch

        x = global_mean_pool(x, batch1)  
        x2 = global_mean_pool(x2, batch2)

        x = torch.cat((x, x_norm2_1), dim = 1)
        x2 = torch.cat((x2, x_norm2_2), dim = 1)

        x = self.lin(x)
        x2 = self.lin(x2)

        x = torch.cat((x, x2), dim = 1)

        # 3. Apply a final classifier

        x = self.lin1(x)
        x = self.lin2(x)
        x = self.softmax(x)
    
        return x
    
class train_utils:
    def train(self, train_loader, train_loader2, model, target, optimizer, scheduler, criterion, edge_col_name):

        model.train()
        
        losses = []

        for data, data2 in zip(train_loader, train_loader2):  
            if edge_col_name==None:
                out = model(data.x_norm, data2.x_norm, data.edge_index, data2.edge_index, data.batch)
            else:
                out =  model(data.x_norm, data2.x_norm, data.edge_index, data2.edge_index, data.batch, data.y, data.x_norm2, data2.x_norm2, data[edge_col_name], data2[edge_col_name])

            loss = criterion(out, data.y[:,target].long())   
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
            losses.append(loss.cpu().detach().numpy())
        scheduler.step()
            
        return np.mean(losses)

    def test(self, loader, loader2, model, target, edge_col_name, metric = metrics.accuracy_score):
        
        model.eval()
        
        score = 0
        for data, data2 in zip(loader, loader2):
            if edge_col_name==None:
                out = model(data.x_norm, data2.x_norm, data.edge_index, data2.edge_index, data.batch)
            else:
                out = model(data.x_norm, data2.x_norm, data.edge_index, data2.edge_index, data.batch, data.y, data.x_norm2, data2.x_norm2, data[edge_col_name], data2[edge_col_name])
                
            y_pred = out.argmax(dim=1).cpu().detach().numpy()
            y_true = data.y[:,target].long().cpu().detach().numpy()
            
            score += metric(y_true, y_pred)
        
        return score/len(loader)

    def train_test(self, device, train_loader, train_loader2, test_loader, test_loader2, model, class_weights, target, edge_col_name=None, 
        learning_rate=0.01, e_patience = 10, min_acc= 0.005, n_epochs=500):
        t0 = time.time()

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                            lr_lambda=lambda epoch: 0.99 ** epoch,
                                            last_epoch=-1,
                                            verbose=False)
        criterion = torch.nn.CrossEntropyLoss(weight = class_weights)

        k=0
        
        loss, train_score, test_score = [], [], []
        
        for epoch in tqdm(range(n_epochs)):
            loss += [self.train(train_loader, train_loader2, model, target, optimizer, scheduler, criterion, edge_col_name)]
            train_score += [self.test(train_loader, train_loader2, model, target, edge_col_name)]
            test_score += [self.test(test_loader, test_loader2, model, target, edge_col_name)]
            
            if (epoch+1)%10==0:
                print(f'Epoch: {epoch+1:03d}, Loss: {loss[-1]:.4f}, Train: {train_score[-1]:.4f}, Test: {test_score[-1]:.4f}')
                print("\nlr", optimizer.param_groups[0]['lr'])
            results = pd.DataFrame({
                'loss': loss,
                'train_score': train_score, 'test_score': test_score,
                'time':(time.time()-t0)/60
            })

            # enable early stopping
            if (epoch > 1) and abs(loss[-1]/loss[-2]-1) < min_acc :
                k += 1
            if k> e_patience:
                print('Early stopping, epoch', epoch)
                break

        return model, results