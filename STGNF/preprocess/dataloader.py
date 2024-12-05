import torch
import numpy as np
import torch.utils.data
import sys
sys.path.append('../preprocess/add_window')
sys.path.append('../data_store')
#sys.path.append('../../preprocessing')
from add_window import Add_Window_Horizon
from load_dataset import load_st_dataset,load_st_dataset_withtimevalue
from normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler

def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler

def split_data_by_days(data, val_days, test_days, interval=60):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    val_data = data[-T*(test_days + val_days): -T*test_days]
    train_data = data[:-T*(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    # print(X.shape)
    # print(Y.shape)
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)

    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)

    return dataloader

def data_loader_dic(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    l=[]
    d=dict()
    print(X.shape)
    print(Y.shape)
    for i in range(0,X.shape[0]):
        d['past_target_norm']=X[i]
        d['future_target_norm']=Y[i]
        l.append(d)
    # dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
    #                                          shuffle=shuffle, drop_last=drop_last)
    print(len(l))

    dataloader = torch.utils.data.DataLoader(l, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    num=0
    for i in dataloader:
        num+=1
        # print(i["past_target_norm"].shape)  #(b,t1,n)
        # print(i["future_target_norm"].shape) #(b,t2,n)
    print(num)
    return dataloader

def get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True):
    #load raw st dataset
    data = load_st_dataset(args.dataset)        # T, N, D
    #normalize st data
    data=data[:,:20,:]
    #print(data.shape)
    data, scaler = normalize_dataset(data, normalizer, args.column_wise)
    #spilit dataset by days or by ratio
    if args.test_ratio > 1:
        data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
    else:
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
    single=False
    #add time window
    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)
    ##############get dataloader######################

    train_dataloader = data_loader(x_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    # for i in train_dataloader:
    #     print(i['history'].shape)
    #     print(i['future'].shape)
    #     break
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, scaler



def get_all_nomask_data_time3(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True):

    data,time = load_st_dataset_withtimevalue(args.dataset)

    data=torch.Tensor(data)  #(t,n,v)
    time=torch.Tensor(time)  #(t,c) 星期,小时,分钟,一天,一周
    time,len_week,len_xiuxi=process_time(args.dataset,time)
    mask=torch.ones(data.shape[0],data.shape[1],data.shape[2])
    data=data[:,:,:3]
    mask=mask[:,:,:3]

    data=data.reshape(-1,data.shape[1]*data.shape[2])  #(T,nv)
    mask = mask.reshape(-1,mask.shape[1]*mask.shape[2]) #(T,nv)
    print(data.shape)
    scaler=0
    std=0
    mean=0

    if args.test_ratio > 1:
        data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
        mask_train, mask_val, mask_test = split_data_by_days(mask, args.val_ratio, args.test_ratio)
        time_train, time_val, time_test = split_data_by_days(time, args.val_ratio, args.test_ratio)
    else:
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
        mask_train, mask_val, mask_test = split_data_by_ratio(mask, args.val_ratio, args.test_ratio)
        time_train, time_val, time_test = split_data_by_ratio(time, args.val_ratio, args.test_ratio)
    single=False

    print(data_train.shape)  #(t,nv)

    std, mean = torch.std_mean(data_train, dim=0)
    std = std.clamp(min=1e-8)
    print(std.shape)
    print(mean.shape)
    return data_train, data_val, data_test, scaler,mask_train,mask_val,mask_test,std,mean,time_train,time_val,time_test,len_week,len_xiuxi

def process_time(dataset,time):  #(t,5)

    #dataset='PEMS04'
    if dataset=='PEMSD4':

        #70 65
        week_list = torch.tensor([0, 7, 10, 13, 16, 19, 22, 28, 34, 40, 43, 49, 55, 61, 64,
                                  67, 70, 73, 79, 85, 88, 91, 94, 97, 100, 103, 106, 109, 113,
                                  117, 120, 126, 130, 137, 140, 148, 152, 157, 160, 165, 171,
                                  176, 179, 182, 188, 191, 194, 198, 201, 204, 207, 210, 214,
                                  218, 221, 224, 227, 230, 233, 236, 239, 242, 245, 252, 263,
                                  266, 269, 275, 281, 287])  # 间距为2
        xiuxi_list = torch.tensor([0, 6, 10, 13, 19, 43, 54, 61, 65, 69, 75, 80, 88, 91, 94, 100, 104,
                                   107, 111, 118, 122, 125, 129, 133, 137, 141, 146, 149, 154, 157, 160,
                                   163, 166, 169, 172, 175, 178, 181, 184, 187, 190, 193, 196, 199, 204,
                                   211, 215, 218, 221, 225, 229, 232, 235, 239, 244, 247, 253, 257, 262,
                                   265, 268, 273, 280, 283, 287])  # 间距为2

    elif dataset=='PEMSD8':
        # 63 71
        week_list = [0, 6, 15, 18, 26, 29, 36, 39, 42, 48, 54, 57, 60, 66, 72, 75, 78, 83, 86, 89, 92,
                95, 98, 101, 104, 107, 112, 118, 121, 159, 162, 165, 169, 177, 182, 189, 192, 195,
                199, 205, 210, 213, 216, 219, 222, 225, 228, 231, 234, 237, 243, 249, 252, 255, 260,
                264, 267, 270, 273, 276, 279, 282, 285]
        xiuxi_list = [0, 6, 12, 15, 18, 21, 27, 31, 34, 41, 46, 50, 54, 57, 66, 69, 72, 75, 78, 81, 84,
                  87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126, 129, 134, 140, 144,
                  148, 152, 155, 159, 163, 166, 171, 178, 184, 190, 196, 202, 208, 214, 220, 223, 229,
                  233, 236, 239, 243, 246, 249, 255, 260, 264, 267, 270, 274, 280, 283, 286]


        week_list=torch.tensor(week_list)
        xiuxi_list=torch.tensor(xiuxi_list)


    len_week=len(week_list)
    len_xiuxi=len(xiuxi_list)

    t_week = torch.abs(week_list[:, None] - time[:,3][None, :])
    t_week = torch.argmin(t_week, dim=0)

    t_xiuxi = torch.abs(xiuxi_list[:, None] - time[:, 3][None,:])
    t_xiuxi = torch.argmin(t_xiuxi, axis=0)
    t_xiuxi=t_xiuxi+len_week

    is_weekend = (time[:,0] == 5) | (time[:,0] == 6)  #周末为1 工作日为0

    t_weekxiuxi=torch.where(is_weekend,t_xiuxi,t_week)

    time=torch.cat([time,is_weekend[:,None],t_weekxiuxi[:,None]],dim=-1)

    return time,len_week,len_xiuxi



