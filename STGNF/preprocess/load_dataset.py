import os
import numpy as np
import sys
import torch
def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD4':
        data_path = os.path.join('../data_store')
        data_path=os.path.join(data_path,'PEMS04')
        data_path=os.path.join(data_path,'pems04.npz')
        #print(data_path)
        sys.path.append(data_path)
        #print(sys.path)
        data = np.load(data_path)['data'][:, :, :]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        # data_path = os.path.join('../data_store')
        # data_path = os.path.join(data_path, 'PEMS08')
        # data_path = os.path.join(data_path, 'pems08.npz')
        data_path = os.path.join('../data_store/PEMS08/pems08.npz')
        sys.path.append(data_path)
        data = np.load(data_path)['data'][:, :, :]  #onley the first dimension, traffic flow data
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data

def load_st_dataset1(dataset):
    #output B, N, D
    if dataset == 'PEMSD4':
        data_path = os.path.join('../../data_store')
        data_path=os.path.join(data_path,'PEMS04')
        data_path=os.path.join(data_path,'pems04.npz')
        #print(data_path)
        sys.path.append(data_path)
        #print(sys.path)
        data = np.load(data_path)['data'][:, :, :]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        # data_path = os.path.join('../data_store')
        # data_path = os.path.join(data_path, 'PEMS08')
        # data_path = os.path.join(data_path, 'pems08.npz')
        data_path = os.path.join('../data_store/PEMS08/pems08.npz')
        sys.path.append(data_path)
        data = np.load(data_path)['data'][:, :, :]  #onley the first dimension, traffic flow data
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data



def get_timesteps():
    import pandas as pd
    # 创建一个时间戳序列
    timestamps = pd.date_range(start='2018-01-01 00:00:00', end='2018-02-28 23:59:59', freq='5T')
    #timestamps = pd.date_range(start='2016-07-01 00:00:00', end='2016-08-31 23:59:59', freq='5T')

    # 提取星期、小时、分钟信息
    weekday = timestamps.weekday.values.reshape(-1, 1)
    hour = timestamps.hour.values.reshape(-1, 1)
    minute = timestamps.minute.values.reshape(-1, 1)

    index = hour * 12 + minute // 5  # 一天有24小时，每小时分为12个五分钟
    all_index=weekday*288+index

    dayminute_onehot = np.eye(288)[index]
    # 将星期、小时、分钟信息转换为 one-hot 编码
    weekday_onehot = np.eye(7)[weekday]
    hour_onehot = np.eye(24)[hour]
    minute_onehot = np.eye(12)[minute // 5]
    is_weekend = (weekday == 5) | (weekday == 6)

    # 将星期、小时、分钟信息的 one-hot 编码拼接在一起
    time_feat = np.concatenate([weekday_onehot, is_weekend.astype(int)[:, :, None], hour_onehot, minute_onehot], axis=2)
    #time_feat=np.concatenate([weekday, hour, minute, index,all_index], axis=-1)


    time_feat = time_feat.squeeze()
    print(time_feat.shape)
    #print(time_feat[:100])
    data_path = os.path.join('../data_store')
    data_path = os.path.join(data_path, 'PEMS04')
    data_path = os.path.join(data_path, 'timestamp2.npz')
    np.savez(data_path, data=time_feat)


def load_st_dataset_withtimevalue(dataset):
    # output B, N, D
    if dataset == 'PEMSD4':
        data_path = os.path.join('../data_store')
        data_path = os.path.join(data_path, 'PEMS04')
        data_path1 = os.path.join(data_path, 'pems04.npz')

        sys.path.append(data_path1)

        data = np.load(data_path1)['data'][:, :, :]  # onley the first dimension, traffic flow data


        data_path_time = os.path.join(data_path, 'timestamp2.npz')
        time1 = np.load(data_path_time)
        time = time1['data'][:, :]

    elif dataset == 'PEMSD8':

        data_path = os.path.join('../data_store/PEMS08')
        data_path1 = os.path.join(data_path, 'pems08.npz')
        sys.path.append(data_path1)
        data = np.load(data_path1)['data'][:, :, :]  # onley the first dimension, traffic flow data

        data_path_time = os.path.join(data_path, 'timestamp2.npz')
        time1 = np.load(data_path_time)
        time = time1['data'][:, :]
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))

    return data,  time  #( t,n,v)(t,c) 星期,小时,分钟,一天


