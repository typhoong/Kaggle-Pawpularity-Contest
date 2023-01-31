# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] id="HB6tcaFW_H3T" papermill={"duration": 0.022571, "end_time": "2021-11-28T13:25:02.216142", "exception": false, "start_time": "2021-11-28T13:25:02.193571", "status": "completed"} tags=[]
# fork https://www.kaggle.com/nishantrajadhyaksha/pawpularity-pytorchlightning-w-b

# + id="qwvvNvo_2LH5" papermill={"duration": 9.927229, "end_time": "2021-11-28T13:25:12.217576", "exception": false, "start_time": "2021-11-28T13:25:02.290347", "status": "completed"} tags=[]
import os
import warnings
from pprint import pprint
from glob import glob
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as T
from box import Box
from timm import create_model
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
import optuna, copy

import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.progress import ProgressBarBase
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningDataModule, LightningModule
# from pytorch_lightning.loggers import WandbLogger
import albumentations as A
from albumentations.pytorch import ToTensorV2
# import wandb
import cv2
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import shutil
import pickle
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import TensorDataset
import time
import gc
warnings.filterwarnings("ignore")
# -

# # Meta info

# + id="spieGAU32LH6" papermill={"duration": 0.02556, "end_time": "2021-11-28T13:25:12.306938", "exception": false, "start_time": "2021-11-28T13:25:12.281378", "status": "completed"} tags=[]
drive_root = '..'
TRAIN_DIR = f"{drive_root}/input/petfinder-pawpularity-score/train"
TEST_DIR = f"{drive_root}/input/petfinder-pawpularity-score/test"
DENSE_FEATURES = [
    'Subject Focus',
    'Eyes',
    'Face',
    'Near',
    'Action',
    'Accessory',
    'Group',
    'Collage',
    'Human',
    'Occlusion',
    'Info',
    'Blur',
    'img_long_axis'
]
df_train = pd.read_csv(f"{drive_root}/input/petfinder-pawpularity-score/train.csv")
df_test = pd.read_csv(f"{drive_root}/input/petfinder-pawpularity-score/test.csv")
df_train['filepath'] = df_train.Id.apply(lambda x :f"{TRAIN_DIR}/{x}.jpg" )
df_test['filepath'] = df_test.Id.apply(lambda x :f"{TEST_DIR}/{x}.jpg" )

long_axis_df_path = f"{drive_root}/input/pet-train-long-axis/df_train_w_long_axis.csv"
long_axis_max = 1280
if os.path.isfile(long_axis_df_path):
    df_train = pd.read_csv(long_axis_df_path)
else:
    df_train['img_long_axis'] = df_train.Id.apply(lambda x : max(cv2.imread(f"{TRAIN_DIR}/{x}.jpg" ).shape[:2]))
    df_train['img_long_axis'] /=long_axis_max
    df_train['img_long_axis'] = df_train['img_long_axis'].astype(np.float32)
    df_train['filepath'] = df_train.Id.apply(lambda x :f"{TRAIN_DIR}/{x}.jpg" )
    
df_test['img_long_axis'] = df_test.Id.apply(lambda x : max(cv2.imread(f"{TEST_DIR}/{x}.jpg" ).shape[:2]))
df_test['img_long_axis'] /=long_axis_max
df_test['img_long_axis'] = df_test['img_long_axis'].astype(np.float32)
df_test['filepath'] = df_test.Id.apply(lambda x :f"{TEST_DIR}/{x}.jpg" )


# -

# # Loss

# + id="8cg0XTpW94Hi" papermill={"duration": 0.024159, "end_time": "2021-11-28T13:25:12.393156", "exception": false, "start_time": "2021-11-28T13:25:12.368997", "status": "completed"} tags=[]
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss


# -

# # Dataset

# + id="5TZT70L4pi4u" papermill={"duration": 0.027435, "end_time": "2021-11-28T13:25:12.623895", "exception": false, "start_time": "2021-11-28T13:25:12.596460", "status": "completed"} tags=[]
class CustomDataset(Dataset):
    def __init__(self, x, y=None):#, image_size=224):
        self._X = x
        self._y = y

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        feature = self._X[idx]
        if self._y is not None:
            label = self._y[idx]
            return feature, label
        return feature

class PetfinderDataModule(LightningDataModule):
    def __init__(
        self,
        train_features, train_y, val_features, val_y, cfg,):
        super().__init__()
        self._train_features = train_features
        self._val_features = val_features
        self._train_y = train_y
        self._val_y = val_y
        self._cfg = cfg

    def __create_dataset(self, train=True):
        if train==True:
            return CustomDataset(self._train_features, self._train_y)
        else:
            return CustomDataset(self._val_features, self._val_y)

    def train_dataloader(self):
        dataset = self.__create_dataset(True)
        return DataLoader(dataset, **self._cfg.train_loader)

    def val_dataloader(self):
        dataset = self.__create_dataset(False)
        return DataLoader(dataset, **self._cfg.val_loader)
    


# +
class EarlyStopper():
    def __init__(self, patience: int, mode:str)-> None:
        self.patience = patience
        self.mode = mode

        # Initiate
        self.patience_counter = 0
        self.stop = False
        self.best_loss = np.inf
        
    def check_early_stopping(self, loss: float)-> None:
        loss = -loss if self.mode == 'max' else loss  # get max value if mode set to max

        if loss >= self.best_loss:
            # got better score
            self.patience_counter += 1
            
            if self.patience_counter == self.patience:
                self.stop = True  # end

        elif loss < self.best_loss:
            # got worse score
            self.patience_counter = 0
            self.best_loss = loss
            


# + id="5FC8MKiq2LH7" papermill={"duration": 0.031668, "end_time": "2021-11-28T13:25:12.674744", "exception": false, "start_time": "2021-11-28T13:25:12.643076", "status": "completed"} tags=[]
config = {'seed': 2025,
          'root': f"{drive_root}", 
          'n_splits': 10,
          'epoch': 1000,#30,
          'patience':10,
          'trainer': {
              'gpus': 1,
              'auto_lr_find':False,
              'accumulate_grad_batches': 1,
              'progress_bar_refresh_rate': 1,
              'fast_dev_run': False,
              'num_sanity_val_steps': 0,
              'resume_from_checkpoint': None,
#               'accelerator':'ddp',
#               'strategy':'ddp'
          },
          'transform':{
              'name': 'albu',
              'image_size': 384,#224,#384,
              'squared':True,
              'aug':'rotate,shift,scale,perspective,more',
          },
          'train_loader':{
              'batch_size': 512,#16,#64,
              'shuffle': True,
              'num_workers': 1,
              'pin_memory': False,
              'drop_last': True,
            
          },
          'val_loader': {
              'batch_size': 512,#16,#64,
              'shuffle': False,
              'num_workers': 8,
              'pin_memory': False,
              'drop_last': False
         },
          'save_discript':'Sigmoid_Bayesian',
          'model':{
              'name': 'swin_large_patch4_window12_384_in22k',#'deit_base_distilled_patch16_384',#'swin_tiny_patch4_window7_224',#
              'img_feature_dim':128,
              'output_dim': 1,
              'first_drop': 0.5,
              'second_drop': 0.5,
              'third_drop': 0.5,
              'last_drop':0.5,
              'activation':'nn.SELU',
              'quantile_num':512,
          },
          'optimizer':{
              'name': 'optim.AdamW',
              'AdamW':{
                  'lr': 1e-5,
                  'betas': (1,1),
                  'weight_decay': 0.01,
                  'amsgrad': False,

                },
          },
          'scheduler':{
              'name': 'optim.lr_scheduler.CosineAnnealingWarmRestarts',
              'CosineAnnealingWarmRestarts':{
                  'T_0': 4,
                  'T_mult':2,
                  'eta_min': 1e-7,
              }
          },
          'loss': 'QuantileLoss',#'nn.BCEWithLogitsLoss',
          'metric': 'RMSELoss',

          'svr':{
              'C':0.5,
          },
          'quantile':{
              'quantile_num':500,
              'batch_size':1024,
              'epoch':1000,
          },
          'svr_train':False,
          'swin_train':False
        
}

config = Box(config)
# -

# # Setting arguments

# + executionInfo={"elapsed": 17, "status": "ok", "timestamp": 1637508005015, "user": {"displayName": "Daewoo Myoung", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhyUBzKneZcIle1t7HXIAaL548lK9ndvDFCUhROwQ=s64", "userId": "15525163033989858669"}, "user_tz": -540} id="b-nCf5Ze-6im" outputId="554d312b-fec4-402a-bb40-a442a7824d56" papermill={"duration": 0.029591, "end_time": "2021-11-28T13:25:12.872125", "exception": false, "start_time": "2021-11-28T13:25:12.842534", "status": "completed"} tags=[]
torch.autograd.set_detect_anomaly(True)
seed_everything(config.seed)

os.environ["CUDA_VISIBLE_DEVICES"]="1"
model_code = 'swin_2025'
import random
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(config.seed)
random.seed(config.seed)

criterion = nn.BCEWithLogitsLoss()
skf = StratifiedKFold(
    n_splits=config.n_splits, shuffle=True, random_state=config.seed
)
activation_list = ['nn.LogSigmoid', 'nn.Hardswish',  'nn.PReLU', 'nn.ReLU', 'nn.ELU', 'nn.ReLU6', 'nn.SELU', 'nn.CELU']

# -

def objective(trial):
    swin_scores=[]
    val_scores=[]
    config.optimizer.AdamW.lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    config.model.h1 = trial.suggest_categorical('h1', [32, 64, 128, 256, 512, 1024, 2048])
    config.model.h2 = trial.suggest_categorical('h2', [32, 64, 128, 256, 512, 1024, 2048])  
    config.patience =  trial.suggest_categorical('patience', [5, 10, 20, 30, 50])
    config.model.activation = trial.suggest_categorical('activation', activation_list)
    config.model.first_drop = trial.suggest_float('first_drop', 0, 0.9)
    config.model.second_drop = trial.suggest_float('second_drop', 0, 0.9)
    config.model.third_drop = trial.suggest_float('third_drop', 0, 0.9)
    
    beta1 = trial.suggest_float('beta1', 0, 1)
    beta2 = trial.suggest_float('beta2', 0, 1)
    threshold = 0.5
    binary_th = trial.suggest_int('binary_th', 80, 100)
    config.optimizer.AdamW.betas = (beta1, beta2)
    config.optimizer.AdamW.weight_decay = trial.suggest_float('weight_decay', 0, 1)
    config.optimizer.AdamW.amsgrad = trial.suggest_categorical('amsgrad', [True, False])
    config.scheduler.CosineAnnealingWarmRestarts.T_0 = trial.suggest_int('T_0', 1, 100)
    config.scheduler.CosineAnnealingWarmRestarts.T_mult = trial.suggest_int('T_mult', 1, 100)
    config.scheduler.CosineAnnealingWarmRestarts.eta_min = trial.suggest_loguniform('eta_min', 1e-10, 1e-5)
    df_train['binary'] = df_train.Pawpularity.apply(lambda x : int(x >= binary_th))
    model_save_dir = f'{drive_root}/output/weights/final_weights/{model_code}/'

    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_train["Id"], df_train["Pawpularity"])):
        early_stopper = EarlyStopper(patience = config.patience, mode='min')
        val_min_score = 1e+100
        train_df = df_train.loc[train_idx].reset_index(drop=True)
        val_df = df_train.loc[val_idx].reset_index(drop=True)

        train_ratio = 100 * sum(df_train['binary'])/len(df_train['binary'])
        val_ratio = 100 * sum(val_df['binary'])/len(val_df['binary'])

        # Load embed features
        val_predicts = np.load(f'{model_save_dir}val_predicts_fold{fold}.npy')
        train_embed_features = np.load(f'{model_save_dir}train_embed_org_fold{fold}.npy')
        val_embed_features = np.load(f'{model_save_dir}val_embed_org_fold{fold}.npy')
        train_embed_flip_features = np.load(f'{model_save_dir}train_embed_flip_fold{fold}.npy')
        val_embed_flip_features = np.load(f'{model_save_dir}val_embed_flip_fold{fold}.npy')
        train_embed_merge_features = np.load(f'{model_save_dir}train_embed_merge_org_fold{fold}.npy')
        val_embed_merge_features = np.load(f'{model_save_dir}val_embed_merge_org_fold{fold}.npy')
        train_embed_merge_flip_features = np.load(f'{model_save_dir}train_embed_merge_flip_fold{fold}.npy')
        val_embed_merge_flip_features = np.load(f'{model_save_dir}val_embed_merge_flip_fold{fold}.npy')
        swin_score = mean_squared_error(val_df.Pawpularity,val_predicts)**0.5
        swin_scores.append(swin_score)

        train_concat_features = np.concatenate((train_embed_merge_features, train_embed_merge_flip_features), axis=0)
        train_loader = DataLoader(CustomDataset(train_concat_features, pd.concat([train_df['binary'], train_df['binary']]).reset_index(drop=True)), shuffle=config.train_loader.shuffle, batch_size=config.train_loader.batch_size)

        # Model
        q = nn.Sequential(
                nn.Dropout(config.model.first_drop), 
                nn.Linear(train_concat_features.shape[1], config.model.h1),
                eval(config.model.activation)(),
                nn.Dropout(config.model.second_drop), 
                nn.Linear(config.model.h1, config.model.h2),
                eval(config.model.activation)(),
                nn.Dropout(config.model.third_drop), 
                nn.Linear(config.model.h2, 1)
        ).to('cuda')

        # Optimizer & scheduler
        optimizer = eval(config.optimizer.name)(q.parameters(), **config.optimizer.AdamW)
        scheduler = eval(config.scheduler.name)(optimizer, **config.scheduler.CosineAnnealingWarmRestarts)

        for epoch in range(config.epoch):
            q.train()
            train_loss = 0
            for feature, label in train_loader:
            
                optimizer.zero_grad()
            
                feature, label = feature.float().to('cuda') , label.float().to('cuda')
            
                pred = q(feature)
            
                loss = criterion(pred.squeeze(dim=-1), label)
            
                loss.backward()
            
                optimizer.step()
            
                scheduler.step()
            
                train_loss += loss.item()
            
            q.eval()
            with torch.no_grad():
                org_feature = torch.FloatTensor(val_embed_merge_features).to('cuda')
                flip_feature = torch.FloatTensor(val_embed_merge_flip_features).to('cuda')
                org_pred = q(org_feature)
                flip_pred = q(flip_feature)
                org_pred = (torch.sigmoid(org_pred).ge(threshold).int().cpu().squeeze(dim=-1)).numpy()
                flip_pred = (torch.sigmoid(flip_pred).ge(threshold).int().cpu().squeeze(dim=-1)).numpy()
            label = val_df['binary'].to_numpy().astype(int)
            pred_arr = (org_pred + flip_pred) == 2
            val_cm = confusion_matrix(label, pred_arr.astype(int))
            val_f1_score = f1_score(label,pred_arr.astype(int), average='micro')
            
            new_val_predicts = copy.copy(val_predicts)
            new_val_predicts[pred_arr] = 100
            val_rmse = mean_squared_error(val_df.Pawpularity, new_val_predicts)**0.5
            TNR = 100 * val_cm[0][0] / (val_cm[0][0] + val_cm[0][1])
            TPR = 100 * val_cm[1][1] / (val_cm[1][0] + val_cm[1][1])

            early_stopper.check_early_stopping(loss=val_rmse)

            if early_stopper.patience_counter == 0:
                min_TNR = TNR
                min_TPR = TPR
                min_f1 = val_f1_score
                val_min_score = val_rmse
                print(f'fold {fold} epoch {epoch} train_ratio : {train_ratio:.4f} TNR : {TNR:.4f} TPR : {TPR:.4f}, F1 : {val_f1_score:.4f}, rmse : {val_rmse:.4f}, orig_rmse : {swin_score:.4f}', end = '\r')
            #    recorder.save_weight(epoch=epoch_index)

            if early_stopper.stop == True:
                break
        trial.report(val_min_score, fold)
            
        if trial.should_prune():
            raise optuna.TrialPruned()
        val_scores.append(val_min_score)
        print(f'fold {fold} epoch {epoch} train_ratio : {train_ratio:.4f} TNR : {min_TNR:.4f} TPR : {min_TPR:.4f}, F1 : {min_f1:.4f}, rmse : {val_min_score:.4f}, orig_rmse : {swin_score:.4f}')
    
    return sum(val_scores)/len(val_scores)

# # Study Object

# + tags=[]
study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=200)
study_save_dir = f'{drive_root}/output/weights/final_weights/{model_code}_2nd_head/binary_head/'

with open(f"{study_save_dir}04-binary-head.pkl", 'wb') as file:
    pickle.dump(study, file)

trial = study.best_trial
print(f"Best trial:{trial.number}")

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
# -

# # Train & Save Best Model

# +
study_save_dir = f'{drive_root}/output/weights/final_weights/{model_code}_2nd_head/binary_head/'

with open(f"{study_save_dir}04-binary-head.pkl", 'rb') as file:
    study = pickle.load(file)
trial = study.best_trial

# +
swin_scores=[]
val_scores=[]
config.optimizer.AdamW.lr = trial.params['learning_rate']
config.model.h1 = trial.params['h1']
config.model.h2 = trial.params['h2']
config.patience =  trial.params['patience']  
config.model.activation = trial.params['activation']
config.model.first_drop = trial.params['first_drop']
config.model.second_drop = trial.params['second_drop'] 
config.model.third_drop = trial.params['third_drop'] 

beta1 = trial.params['beta1']  
beta2 = trial.params['beta2'] 
threshold = 0.5
binary_th = trial.suggest_int('binary_th', 80, 100)
config.optimizer.AdamW.betas = (beta1, beta2)
config.optimizer.AdamW.weight_decay = trial.params['weight_decay']  
config.optimizer.AdamW.amsgrad = trial.params['amsgrad'] 
config.scheduler.CosineAnnealingWarmRestarts.T_0 = trial.params['T_0']
config.scheduler.CosineAnnealingWarmRestarts.T_mult = trial.params['T_mult'] 
config.scheduler.CosineAnnealingWarmRestarts.eta_min = trial.params['eta_min']
model_save_dir = f'{drive_root}/output/weights/final_weights/{model_code}/'

for fold, (train_idx, val_idx) in enumerate(skf.split(df_train["Id"], df_train["Pawpularity"])):
    early_stopper = EarlyStopper(patience = config.patience, mode='max')
    val_min_score = 1e+100
    binary_dir = f'{drive_root}/output/weights/final_weights/{model_code}_2nd_head/binary_head/Binary_fold{fold}.ckpt'
    train_df = df_train.loc[train_idx].reset_index(drop=True)
    val_df = df_train.loc[val_idx].reset_index(drop=True)
    
    train_ratio = 100 * sum(df_train['binary'])/len(df_train['binary'])
    val_ratio = 100 * sum(val_df['binary'])/len(val_df['binary'])
    
    # Load embed features
    val_predicts = np.load(f'{model_save_dir}val_predicts_fold{fold}.npy')
    train_embed_features = np.load(f'{model_save_dir}train_embed_org_fold{fold}.npy')
    val_embed_features = np.load(f'{model_save_dir}val_embed_org_fold{fold}.npy')
    train_embed_flip_features = np.load(f'{model_save_dir}train_embed_flip_fold{fold}.npy')
    val_embed_flip_features = np.load(f'{model_save_dir}val_embed_flip_fold{fold}.npy')
    train_embed_merge_features = np.load(f'{model_save_dir}train_embed_merge_org_fold{fold}.npy')
    val_embed_merge_features = np.load(f'{model_save_dir}val_embed_merge_org_fold{fold}.npy')
    train_embed_merge_flip_features = np.load(f'{model_save_dir}train_embed_merge_flip_fold{fold}.npy')
    val_embed_merge_flip_features = np.load(f'{model_save_dir}val_embed_merge_flip_fold{fold}.npy')
    train_concat_features = np.concatenate((train_embed_merge_features, train_embed_merge_flip_features), axis=0)


    swin_score = mean_squared_error(val_df.Pawpularity,val_predicts)**0.5
    swin_scores.append(swin_score)
    train_loader = DataLoader(CustomDataset(train_concat_features, pd.concat([train_df['binary'], train_df['binary']]).reset_index(drop=True)), shuffle=config.train_loader.shuffle, batch_size=config.train_loader.batch_size)
    
    # Model
    q = nn.Sequential(
            nn.Dropout(config.model.first_drop), 
            nn.Linear(train_concat_features.shape[1], config.model.h1),
            eval(config.model.activation)(),
            nn.Dropout(config.model.second_drop), 
            nn.Linear(config.model.h1, config.model.h2),
            eval(config.model.activation)(),
            nn.Dropout(config.model.third_drop), 
            nn.Linear(config.model.h2, 1)
    ).to('cuda')
        
    # Optimizer & scheduler
    optimizer = eval(config.optimizer.name)(q.parameters(), **config.optimizer.AdamW)
    scheduler = eval(config.scheduler.name)(optimizer, **config.scheduler.CosineAnnealingWarmRestarts)

    
    for epoch in range(config.quantile.epoch):
        q.train()
        train_loss = 0
        for feature, label in train_loader:
            optimizer.zero_grad()
            feature, label = feature.float().to('cuda') , label.float().to('cuda')
            pred = q(feature)
            loss = criterion(pred.squeeze(dim=-1), label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        q.eval()
        with torch.no_grad():
            org_feature = torch.FloatTensor(val_embed_merge_features).to('cuda')
            flip_feature = torch.FloatTensor(val_embed_merge_flip_features).to('cuda')
            org_pred = q(org_feature)
            flip_pred = q(flip_feature)
            org_pred = (torch.sigmoid(org_pred).ge(threshold).int().cpu().squeeze(dim=-1)).numpy()
            flip_pred = (torch.sigmoid(flip_pred).ge(threshold).int().cpu().squeeze(dim=-1)).numpy()
        label = val_df['binary'].to_numpy().astype(int)
        pred_arr = (org_pred + flip_pred) == 2
        val_cm = confusion_matrix(label, pred_arr.astype(int))
        val_f1_score = f1_score(label,pred_arr.astype(int), average='micro')
        new_val_predicts = copy.copy(val_predicts)
        new_val_predicts[pred_arr] = 100
        val_rmse = mean_squared_error(val_df.Pawpularity, new_val_predicts)**0.5
        TNR = 100 * val_cm[0][0] / (val_cm[0][0] + val_cm[0][1])
        TPR = 100 * val_cm[1][1] / (val_cm[1][0] + val_cm[1][1])

        early_stopper.check_early_stopping(loss=val_rmse)
        if early_stopper.patience_counter == 0:
            pickle.dump(q, open(binary_dir, "wb"))
            min_TNR = TNR
            min_TPR = TPR
            min_f1 = val_f1_score
            val_min_score = val_rmse
            print(f'fold {fold} epoch {epoch} train_ratio : {train_ratio:.4f} TNR : {TNR:.4f} TPR : {TPR:.4f}, F1 : {val_f1_score:.4f}, rmse : {val_rmse:.4f}, orig_rmse : {swin_score:.4f}', end = '\r')

        if early_stopper.stop == True:
            break
    #val_scores.append(val_min_score)
    print(f'fold {fold} epoch {epoch} train_ratio : {train_ratio:.4f} TNR : {min_TNR:.4f} TPR : {min_TPR:.4f}, F1 : {val_min_score:.4f}')
# -

# # Make prediction file

# +

skf = StratifiedKFold(
    n_splits=config.n_splits, shuffle=True, random_state=config.seed
)
orig_scores = []
scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(df_train["Id"], df_train["Pawpularity"])):
    model_save_dir = f'{drive_root}/output/weights/final_weights/{model_code}/'
    head_dir = f'{drive_root}/output/weights/final_weights/{model_code}_2nd_head/binary_head/Binary_fold{fold}.ckpt'
    result_save_dir = f'{drive_root}/output/weights/final_weights/{model_code}_2nd_head/binary_head/'

    train_df = df_train.loc[train_idx].reset_index(drop=True)
    val_df = df_train.loc[val_idx].reset_index(drop=True)

    # Load embed features
    val_predicts = np.load(f'{model_save_dir}val_predicts_fold{fold}.npy')
    val_embed_features = np.load(f'{model_save_dir}val_embed_org_fold{fold}.npy')
    val_embed_flip_features = np.load(f'{model_save_dir}val_embed_flip_fold{fold}.npy')
    val_embed_merge_features = np.load(f'{model_save_dir}val_embed_merge_org_fold{fold}.npy')
    val_embed_merge_flip_features = np.load(f'{model_save_dir}val_embed_merge_flip_fold{fold}.npy')
    
    org_feature = torch.FloatTensor(val_embed_merge_features).to('cuda')
    flip_feature = torch.FloatTensor(val_embed_merge_flip_features).to('cuda')
    q = pickle.load(open(head_dir, "rb"))
    q.eval()
    with torch.no_grad():
        org_pred = q(org_feature)
        flip_pred = q(flip_feature)
        org_pred = (torch.sigmoid(org_pred).ge(threshold).int().cpu().squeeze(dim=-1)).numpy()
        flip_pred = (torch.sigmoid(flip_pred).ge(threshold).int().cpu().squeeze(dim=-1)).numpy()
    pred_arr = (org_pred + flip_pred) == 2
    new_val_predicts = copy.copy(val_predicts)
    new_val_predicts[pred_arr] = 100
    val_rmse = mean_squared_error(val_df.Pawpularity, new_val_predicts)**0.5
    print(f'fold {fold} score : {val_rmse}')
    np.save(f'{result_save_dir}val_binary_predicts{fold}.npy', pred_arr)
    scores.append(val_rmse)
cv_score = sum(scores)/len(scores)
print(f'cv score : {cv_score:.4f}')
