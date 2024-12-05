from pts import Trainer
import torch
import argparse
import os
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES']='3'
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


from load_dataset import load_st_dataset,load_st_dataset_withtimevalue

import random

torch.set_num_threads(2)

DATASET='PEMSD4'
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
save_path='../reports/pems04/{date:%Y-%m-%d_%H %M %S}'.format(date=now)
checkpoint_dir = save_path+'/checkpoint'
def fix_random_seeds(seed=12):
    torch.manual_seed(seed) #cpu设置
    torch.cuda.manual_seed_all(seed)# gpu设置种子
    np.random.seed(seed)
    random.seed(seed)
fix_random_seeds()

data,time = load_st_dataset_withtimevalue(args.dataset)


flow=data[:,:,0]
print(time[:,0].max(),time[:,1].max(),time[:,2].max(),time[:,3].max(),time[:,4].max())

print(time[14112])
data_flow=data[:14112,:,0]

data_flow_1=data_flow.reshape(-1,7*288,307).mean(0)
data_flow=data_flow.reshape(-1,7,288,307)
data_flow_week=data_flow[:,:5,:,:]
data_flow_67=data_flow[:,5:,:,:]
data_flow_week=data_flow_week.mean((0,1))
data_flow_67=data_flow_67.mean((0,1))



data_occ=data[:14112,:,1]
print(data_occ.max())
print(data_occ.min())
print(data_occ.mean())
print(data_occ.std())
print(np.median(data_occ))

data_occ_1=data_occ.reshape(-1,7*288,307).mean(0)
data_occ=data_occ.reshape(-1,7,288,307)
data_occ_week=data_occ[:,:5,:,:]
data_occ_67=data_occ[:,5:,:,:]
data_occ_week=data_occ_week.mean((0,1))  #(288,307)
data_occ_67=data_occ_67.mean((0,1))   #(288,307)

import matplotlib.pyplot as plt
data_speed=data[:14112,:,2]
print(data_speed.max())
print(data_speed.min())
print(data_speed.mean())
print(data_speed.std())
print(np.median(data_speed))

data_speed_1=data_speed.reshape(-1,7*288,307).mean(0)
data_speed=data_speed.reshape(-1,7,288,307)
data_speed_week=data_speed[:,:5,:,:]
data_speed_67=data_speed[:,5:,:,:]
data_speed_week=data_speed_week.mean((0,1))  #(288,307)
data_speed_67=data_speed_67.mean((0,1))   #(288,307)


import matplotlib.pyplot as plt
def plt_week(data,title): #(7*288,n)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(range(7*288), data[:,5],label='location 5',color="#1594A5")

    ax.plot(range(7 * 288), data[:, 13], label='location 13',color="#ffb26b")  #color="#FDC648"f0cb87


    legend_font = {
        #'weight': 'bold',
        'size': 20,
    }
    ax.legend(loc='upper right',bbox_to_anchor=(1.05, 1.035), prop=legend_font,frameon=False,ncol=2,columnspacing=0.3) #frameon=False,

    x_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

    plt.xticks([i*288+144 for i in range(7)], x_labels)

    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontsize(20) for label in labels]
    ax.set_ylim([0, 527])
    plt.xlabel('day', fontsize=25)
    plt.ylabel('flow', fontsize=25)
    #plt.title(title,fontsize=20)
    plt.title('(a) averaged weekly traffic flow', fontsize=25,x=-0.2,loc='left',y=-0.33)
    plt.savefig('averaged_weekly_traffic_flow.pdf', bbox_inches='tight')
    plt.show()

plt_week(data_flow_1,'flow')
#plt_week(data_speed_1.mean(1),'speed')
#plt_week(data_occ_1.mean(1),'occ')
import numpy as np

def distance(point1, point2, point):
    # 计算点到直线的距离
    x1, y1 = point1
    x2, y2 = point2
    x0, y0 = point

    return np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

def douglas_peucker(points, epsilon):
    if len(points) <= 2:
        return points

    # 找到距离最大的点
    dmax = 0
    index = 0
    end = len(points) - 1
    for i in range(1, end):
        d = distance(points[0], points[end], points[i])
        if d > dmax:
            index = i
            dmax = d
    #print(dmax)
    # 如果最大距离大于阈值，则递归调用算法对两边的点集进行处理
    if dmax > epsilon:
        # 分割点集
        results1 = douglas_peucker(points[:index + 1], epsilon)
        results2 = douglas_peucker(points[index:], epsilon)

        # 合并结果
        return np.vstack((results1[:-1], results2))
    else:
        # 如果最大距离小于等于阈值，则直接返回起始点和终止点
        return np.vstack((points[0], points[end]))


def  plt_day(now_data,title1,window_length = 27,poly_order = 3,epsilon = 2): #(288)

    fig, ax = plt.subplots(figsize=(6, 5))
    now_data=np.concatenate((now_data[-24:],now_data,now_data[:24]),axis=-1)

    from scipy.signal import savgol_filter

    now_data = savgol_filter(now_data, window_length, poly_order)


    now_data1 = np.concatenate((np.arange(0, 336)[:, None], now_data[:, None]), axis=-1)

    # 应用Douglas-Peucker算法
    result = douglas_peucker(now_data1, epsilon)
    result1 = result[:, 0].astype(int)


    # 计算每个时间点的流量累积和
    cumulative_sum = np.cumsum(now_data)
    print(title1)

    #slopes = np.diff(cumulative_sum)
    #print(slopes)
    slopes1=np.diff(now_data)
    slopes11=np.gradient(now_data)
    slopes12 = np.gradient(slopes11)


    from scipy.signal import argrelextrema
    min_indices = argrelextrema(now_data, np.less)[0]
    max_indices = argrelextrema(now_data, np.greater)[0]
    print(min_indices)
    print(max_indices)
    #最大值，最小值
    global_min_index = np.argmin(now_data)
    global_max_index = np.argmax(now_data)
    print(global_min_index)
    print(global_max_index)


    indices1=np.concatenate((min_indices,max_indices,[global_min_index,global_max_index]),axis=-1)
    min_indices=min_indices[min_indices>100]
    if 'speed' in title1:
        print('min')
        indices1 = np.concatenate((indices1, min_indices-6,min_indices+6), axis=-1)
    else:
        print('flow occ')
        indices1 = np.concatenate((indices1, max_indices - 6 ,max_indices + 6), axis=-1)
    indices1=indices1[indices1<336]


    index=[]
    index+=range(0,150,6)
    max_slope_indices1 = np.argsort(np.abs(slopes11))[-150:]
    max_slope_indices1.sort()
    max_slope_indices1=max_slope_indices1[index]

    max_slope_indices2 = np.argsort(np.abs(slopes11))[:100]
    max_slope_indices2.sort()
    max_slope_indices2 = max_slope_indices2[range(0,100,6)]
    print(max_slope_indices2)

    max_slope_indices1.sort()

    index1=np.concatenate(([24],indices1,max_slope_indices1,max_slope_indices2,[311]),axis=-1)


    all_index=np.sort(np.unique(index1))
    #print(all_index.shape)   #峰值10+梯度25
    print(list(all_index))


    # 将整体的面积均分成100份
    total_area = cumulative_sum[-1]
    area_per_section = total_area / 50

    # 找到每个时间点对应的x轴值，使得该时间点的流量累积和最接近当前区间的面积
    x_values = []
    current_area = 0
    for i in range(1, len(cumulative_sum)):
        if cumulative_sum[i] >= current_area + area_per_section:
            x_values.append(i)
            current_area += area_per_section
    x_values.sort()
    #print(x_values)

    x=range(0,288)


    plt.plot(range(0,336), now_data)
    plt.scatter(max_slope_indices1, now_data[max_slope_indices1], color='red', label='largest 1st derivative')
    plt.scatter(max_slope_indices2, now_data[max_slope_indices2], color='green', label='smallest 1st derivative')

    plt.scatter(indices1, now_data[indices1], color='blue', label='traffic peak')

    for t in [24,312]:
       plt.axvline(x=t, color='black', linestyle='--')

    legend_font = {
        # 'weight': 'bold',
        'size': 20,
    }

    if 'flow' in title1:
        attribute1='flow'
    if 'occ' in title1:
        attribute1='occupancy'
    if 'speed' in title1:
        attribute1='speed'

    if 'flow' in title1:
        ax.legend(bbox_to_anchor=(-0.07, 1.06), loc="upper left", prop=legend_font,frameon=False,labelspacing=0.2,columnspacing=0.3) #frameon=False,

    if 'occ' in title1:
        ax.legend(bbox_to_anchor=(0.25, 0.30), loc="upper left", prop=legend_font)

    x_labels = ['22']
    # y_labels = []
    for i in range(0,24,6):
         x_labels.append(str(i))
    #x_labels.append('1')
    x_labels.append('0')
    x_labels.append('2')
    x_labele1=[0,12*2,12*8,12*14,12*20,12*26,12*28]
    plt.xticks(x_labele1, x_labels)

    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontsize(20) for label in labels]
    ax.set_ylim([30, 400])
    plt.xlabel('hour',fontsize=25)
    plt.ylabel(attribute1,fontsize=25)
    #plt.legend()
    if 'flow' in title1:
        title2='(b) daily traffic flow on weekdays'
    if 'occ' in title1:
        title2 = '(c) daily traffic occupancy on weekdays'
    if 'speed' in title1:
        title2 = '(c) daily traffic speed on weekdays'
    plt.title(title2,fontsize=25,x=-0.25,loc='left',y=-0.35)
    plt.savefig('averaged_daily_traffic_'+attribute1+'(1).pdf', bbox_inches='tight')
    # 显示图形
    plt.show()

    return index1

def plt_smooth(data):
    x=range(0,288)

    #移动平均法
    #mean_data =np.convolve(data, np.ones(10)/10, mode='same')

    # 指数加权移动平均法
    # weights = np.exp(np.linspace(-1., 0., 20))
    # weights /= weights.sum()
    # mean_data =np.convolve(data, weights, mode='same')
    # print(mean_data.shape)


    #Loess平滑法
    # from statsmodels.nonparametric.smoothers_lowess import lowess
    # mean_data=lowess(data, range(len(data)), frac=0.5, return_sorted=False)

    #Savitzky-Golay滤波器
    from scipy.signal import savgol_filter
    window_length = 37 # 窗口长度
    poly_order = 1  # 多项式拟合阶数
    mean_data = savgol_filter(data, window_length, poly_order)


    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    model = SimpleExpSmoothing(data)
    fit = model.fit(smoothing_level=0.2)
    trend = fit.level


    #plt.plot(x, data, label='1')
    plt.plot(x, mean_data, label='average')
    #plt.plot(x, trend, label='average1')

    plt.xlabel('time')
    plt.ylabel('flow')
    plt.legend()
    plt.title("average  31")
    # 显示图形
    plt.show()
    return mean_data

flow_week=plt_day(data_flow_week.mean(-1),"flow week",window_length = 29,poly_order = 3,epsilon = 2)
print('flow_week')
print(list(flow_week))

#plt_smooth(data_flow_67.mean(-1))
flow_67=plt_day(data_flow_67.mean(-1),"flow 67",window_length = 27,poly_order = 3,epsilon = 2)

#plt_smooth(data_occ_week.mean(-1))
occ_week=plt_day(data_occ_week.mean(-1),"occ week",window_length = 27,poly_order = 3,epsilon = 0.001)

#plt_smooth(data_occ_67.mean(-1))
occ_67=plt_day(data_occ_67.mean(-1),"occ 67",window_length = 21,poly_order = 3,epsilon = 0.001)

#plt_smooth(data_speed_week.mean(-1))
speed_week=plt_day(data_speed_week.mean(-1),"speed week",window_length = 27,poly_order = 3,epsilon = 0.5)

#plt_smooth(data_speed_67.mean(-1))
speed_67=plt_day(data_speed_67.mean(-1),"speed 67",window_length = 37,poly_order = 1,epsilon = 0.01)


week_set = set(flow_week) | set(occ_week) | set(speed_week)
week_list = list(week_set)
week_list.sort()
print(week_list)
print(len(flow_week),len(occ_week),len(speed_week),len(week_list))
week_list=np.array(week_list)
#
week_list=week_list[week_list>23]
week_list=week_list[week_list<312]
week_list=week_list-24

week_list1=[]
i1=0
week_list1.append(week_list[0])
for i in week_list[1:]:
    if i>i1+2:
        week_list1.append(i)
        i1=i
print("week_list")
print(len(week_list1))
print(week_list1)

def plot_all(data,point,title,window_length, poly_order):
    from scipy.signal import savgol_filter

    data = savgol_filter(data, window_length, poly_order)

    fig, ax = plt.subplots(figsize=(6, 5))

    plt.scatter(point, data[point],color='green', label='important timestamps',zorder=2,s=25)
    plt.plot(range(0, 288), data, zorder=3, label='traffic flow')

    legend_font = {
        # 'weight': 'bold',
        'size': 20,
    }

    if 'flow' in title:
        ax.legend(bbox_to_anchor=(-0.03, 1.05), loc="upper left", prop=legend_font,frameon=False)
    if 'occ' in title:
        ax.legend(bbox_to_anchor=(-0.05, 1.07), loc="upper left", prop=legend_font, frameon=False,labelspacing=0.2,columnspacing=0.3)


    x_labels = []
    # y_labels = []
    for i in range(0, 25, 6):
        x_labels.append(str(i))
    # x_labels.append('1')
    plt.xticks([i * 72 for i in range(5)], x_labels)
    # plt.yticks([0.5, 1.5, 2.5], y_labels)
    # plt.legend()
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    # [label.set_fontweight('bold') for label in labels]
    [label.set_fontsize(20) for label in labels]

    plt.xlabel('hour', fontsize=25)
    plt.ylabel(title, fontsize=25)
    ax.set_ylim([30, 390])
    #ax.set_ylim([0, 0.12])
    # plt.legend()
    if title=='flow':
        title2 = '(c) weekday important timestamps'
        plt.title(title2, fontsize=25, loc='left', x=-0.265, y=-0.33)
    if title=='occupancy':
        title2 = '(c) weekday important timestamps'
        plt.title(title2, fontsize=25, loc='left', x=-0.26, y=-0.33)
    if  title=='speed':
        title2 = '(f) average daily traffic speed on weekdays'
        plt.title(title2, fontsize=20, loc='left', x=-0.20, y=1.02)
    plt.tight_layout()
    plt.savefig('weekday_important_timestamps_traffic_' + title + '.pdf', bbox_inches='tight')
    plt.show()


week_set = set(flow_week) | set(occ_week) | set(speed_week)
week_list = list(week_set)
week_list.sort()
print(week_list)
print(len(flow_week),len(occ_week),len(speed_week),len(week_list))

week_list=np.array(week_list)
week_list=week_list[week_list>23]
week_list=week_list[week_list<312]
week_list=week_list-24
print(len(week_list))
print(list(week_list))

week_list1=[]
i1=0
week_list1.append(week_list[0])
for i in week_list[1:]:
    if i>i1+1:
        week_list1.append(i)
        i1=i
print("week_list")
print(len(week_list1))
print(week_list1)
plot_all(data_flow_week.mean(-1),week_list1,'flow',window_length=29, poly_order=3)

list_67_1=[0, 6, 10, 13, 19, 43, 54, 61, 65, 69, 75, 80, 88, 91, 94, 100, 104,
                                   107, 111, 118, 122, 125, 129, 133, 137, 141, 146, 149, 154, 157, 160,
                                   163, 166, 169, 172, 175, 178, 181, 184, 187, 190, 193, 196, 199, 204,
                                   211, 215, 218, 221, 225, 229, 232, 235, 239, 244, 247, 253, 257, 262,
                                   265, 268, 273, 280, 283, 287]


print("list_67")
set_67 = set(flow_67) | set(occ_67) | set(speed_67)
list_67 = list(set_67)
list_67.sort()
print(list_67)
print(len(flow_67),len(occ_67),len(speed_67),len(list_67))
list_67=np.array(list_67)

list_67=list_67[list_67>23]
list_67=list_67[list_67<312]
list_67=list_67-24
print(len(list_67))
print(list(list_67))

list_67_1=[]
i1=list_67[0]
list_67_1.append(i1)
for i in list_67[1:]:
    if i>i1+1:
        list_67_1.append(i)
        i1=i
print(len(list_67_1))
print(list_67_1)

plot_all(data_flow_67.mean(-1),list_67_1,'flow',window_length = 27,poly_order = 3)
plot_all(data_occ_67.mean(-1),list_67_1,'occ',window_length = 21,poly_order = 3)
plot_all(data_speed_67.mean(-1),list_67_1,'speed',window_length = 37,poly_order = 1)

week=[0, 7, 10, 13, 16, 19, 22, 28, 34, 40, 43, 49, 55, 61, 64,
                                  67, 70, 73, 79, 85, 88, 91, 94, 97, 100, 103, 106, 109, 113,
                                  117, 120, 126, 130, 137, 140, 148, 152, 157, 160, 165, 171,
                                  176, 179, 182, 188, 191, 194, 198, 201, 204, 207, 210, 214,
                                  218, 221, 224, 227, 230, 233, 236, 239, 242, 245, 252, 263,
                                  266, 269, 275, 281, 287]
list67=[0, 6, 10, 13, 19, 43, 54, 61, 65, 69, 75, 80, 88, 91, 94, 100, 104,
                                   107, 111, 118, 122, 125, 129, 133, 137, 141, 146, 149, 154, 157, 160,
                                   163, 166, 169, 172, 175, 178, 181, 184, 187, 190, 193, 196, 199, 204,
                                   211, 215, 218, 221, 225, 229, 232, 235, 239, 244, 247, 253, 257, 262,
                                   265, 268, 273, 280, 283, 287]






