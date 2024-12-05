from pts import Trainer
import torch
import argparse
import os
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES']='0'
import sys
REPO_NAME = "STGNF"
def get_repo_basepath():
    cd = os.path.abspath(os.curdir)
    print(cd)
    return cd[:cd.index(REPO_NAME) + len(REPO_NAME)]
REPO_BASE_PATH = get_repo_basepath()
sys.path.append(REPO_BASE_PATH)
print(REPO_BASE_PATH)

import sys
import numpy as np
sys.path.append('../')
sys.path.append('../preprocess')
sys.path.append('../data_store')
sys.path.append('../stgnf')


from stgnf.model.estimator import STGNFEstimator

from stgnf.model.metrics import compute_validation_metrics
from stgnf.gluon.train import STGNFTrainer

from preprocess.dataloader import get_all_nomask_data_time3

import pandas as pd
import random
from gluonts.dataset.common import DataEntry, Dataset, ListDataset
from preprocess.get_adj import get_adjacency_matrix1


torch.set_num_threads(2)

DATASET='PEMSD8'
if DATASET == 'MetrLA':
    NODE_NUM = 207
elif DATASET == 'BikeNYC':
    NODE_NUM = 128
elif DATASET == 'SIGIR_solar':
    NODE_NUM = 137
elif DATASET == 'SIGIR_electric':
    NODE_NUM = 321
elif DATASET=='PEMSD4':
    NODE_NUM=307
elif DATASET=='PEMSD8':
    NODE_NUM=170

parser = argparse.ArgumentParser(description='PyTorch dataloader')
parser.add_argument('--dataset', default=DATASET, type=str)
parser.add_argument('--num_nodes', default=NODE_NUM, type=int)
parser.add_argument('--val_ratio', default=0.1, type=float)
parser.add_argument('--test_ratio', default=0.2, type=float)
parser.add_argument('--lag', default=12, type=int)
parser.add_argument('--horizon', default=12, type=int)
parser.add_argument('--column_wise', default=False, type=int)
parser.add_argument('--batch_size', default=64, type=int)
args = parser.parse_args()
now=datetime.now()
save_path='../reports/pems08/{date:%Y-%m-%d_%H %M %S}'.format(date=now)
#save_path='../reports/pems04/tactis/{date:2023-02-20_00 54 12}'
checkpoint_dir = save_path+'/checkpoint'
def fix_random_seeds(seed=12):
    torch.manual_seed(seed) #cpu设置
    torch.cuda.manual_seed_all(seed)# gpu设置种子
    np.random.seed(seed)
    random.seed(seed)
fix_random_seeds()


args.val_ratio=0.1
args.test_ratio=0.1

traindata, val, test, scaler,mask_train,mask_val,mask_test,std,mean,time_train,time_val,time_test,len_week,len_xiuxi = get_all_nomask_data_time3(args, normalizer='std', tod=False,dow=False, weather=False, single=True)

print("mask:",mask_train.shape,mask_val.shape,mask_test.shape)
print("time:",time_train.shape,time_val.shape,time_test.shape)
print(time_train[...,0].max(),time_train[...,1].max(),time_train[...,2].max(),time_train[...,3].max(),time_train[...,4].max())


train_data=[]
train=traindata.transpose(1,0)  #(N,T)
mask_train=mask_train.transpose(1,0)  #(N,T)
time_train=time_train.transpose(1,0)
d = dict()
d['start'] = pd.Timestamp('2018-07-01 00:00:00', freq='5T')
d['target']=train
d['feat_dynamic_real']=mask_train
d['feat_dynamic_cat']=time_train
train_data.append(d)
train_data=ListDataset(train_data,freq='5T',one_dim_target = False)

val_data=[]
val=val.transpose(1,0)  #(N,T)
mask_val=mask_val.transpose(1,0)  #(N,T)
time_val=time_val.transpose(1,0)
d = dict()
d['start'] = pd.Timestamp('2018-08-10 00:00:00', freq='5T')
d['target']=val
d['feat_dynamic_real']=mask_val   #(N,T)
d['feat_dynamic_cat']=time_val
val_data.append(d)
val_data=ListDataset(val_data,freq='5T',one_dim_target = False)



test_data=[]
test=test.transpose(1,0)
mask_test=mask_test.transpose(1,0)
time_test=time_test.transpose(1,0)
d = dict()
d['start']= pd.Timestamp('2018-08-20 00:00:00',freq='5T')

d['target']=test
d['feat_dynamic_real'] = mask_test
d['feat_dynamic_cat']=time_test
test_data.append(d)
test_data=ListDataset(test_data,freq='5T',one_dim_target = False)



device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




import pickle
with open('../data_store/PEMS08/n_embedding_pems08.pkl', "rb") as file:
    data_node1 = pickle.load(file)
    data_node1=torch.tensor(data_node1)
    print(data_node1.shape)

estimator = STGNFEstimator(
    model_parameters = {
        "num_variables":3,
        "series_embedding_dim": 12, #10 24
        "input_encoder_layers": 2,
        "input_encoding_normalization": True,

        "loss_normalization": "series",
        "time_num":len_week+len_xiuxi,
        "data_normalization": "standardizationall",
        "mean":mean.reshape(170,3).to(device),
        "std":std.reshape(170,3).to(device),
        "data_node":data_node1.to(device),

        "time_encoding":{
            "dropout": 0.01,
            "input_embedding":44,
        },
        "temporal_encoder":{
            "num_node":args.num_nodes,
            "num_attribute":3,
            "attention_layers": 4,
            "attention_heads": 2,
            "attention_dim": 24,
            "attention_feedforward_dim": 24,
            "dropout": 0.01,
        },
        "copula_decoder":{
            "attention_heads": 3,
            "attention_layers": 1,
            "attention_dim": 8,
            "mlp_layers": 1,
            "attention_feedforward_dim": 12,
            "mlp_dim": 48,
            "dropout":  0.1,
            "n_blocks":6,
            "input_size":3,
            "n_hidden":4,
            "batch_norm":False,

        },

    },
    num_series = NODE_NUM*3,

    history_length=36,  # 3*24
    prediction_length=12,  # 24
    freq='5T',

    use_feat_dynamic_real=True,
    trainer = STGNFTrainer(
        # epochs = 100,
        # batch_size = 64,
        epochs =100,
        train_batch_size =4,
        val_batch_size =4,
        num_batches_per_epoch =512,
        learning_rate = 1e-3,
        weight_decay = 0,
        maximum_learning_rate = 1e-3,
        early_stop_patience=50,
        checkpoint_dir=checkpoint_dir,
        clip_gradient = 1e3,
        device = device,
    ),
    cdf_normalization = False,
    num_parallel_samples = 100,
)


predictor = estimator.train(train_data,val_data) #(N,T)

predictor.batch_size = 8
metrics = compute_validation_metrics(
    predictor=predictor,
    dataset=test_data,
    window_length=estimator.history_length + estimator.prediction_length,
    num_samples=100,
    split=True,
    multi=3,
)
print("test")
print(metrics)
