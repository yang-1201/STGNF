from pts import Trainer
import torch
import argparse
import os
from datetime import datetime



os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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


from load_dataset import load_st_dataset_withtimevalue

import random

torch.set_num_threads(2)

DATASET = 'PEMSD8'
if DATASET == 'MetrLA':
    NODE_NUM = 207
elif DATASET == 'BikeNYC':
    NODE_NUM = 128
elif DATASET == 'SIGIR_solar':
    NODE_NUM = 137
elif DATASET == 'SIGIR_electric':
    NODE_NUM = 321
elif DATASET == 'PEMSD4':
    NODE_NUM = 307
elif DATASET == 'PEMSD8':
    NODE_NUM = 170

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
now = datetime.now()
save_path = '../reports/pems08/{date:%Y-%m-%d_%H %M %S}'.format(date=now)
# save_path='../reports/pems04/tactis/{date:2023-02-20_00 54 12}'
checkpoint_dir = save_path + '/checkpoint'


def fix_random_seeds(seed=12):
    torch.manual_seed(seed)  # cpu设置
    torch.cuda.manual_seed_all(seed)  # gpu设置种子
    np.random.seed(seed)
    random.seed(seed)


fix_random_seeds()
data, time = load_st_dataset_withtimevalue(args.dataset)
print(data.shape)
print(time.shape)



flow = data[:, :, 0]
print(time[:, 0].max(), time[:, 1].max(), time[:, 2].max(), time[:, 3].max(), time[:, 4].max())


data_flow = data[864:14976, :, 0]
print(data_flow.shape)
data_flow_1 = data_flow.reshape(-1, 7 * 288, 170).mean(0)
data_flow = data_flow.reshape(-1, 7, 288, 170)
data_flow_week = data_flow[:, :5, :, :]
data_flow_67 = data_flow[:, 5:, :, :]
data_flow_week = data_flow_week.mean((0, 1))
data_flow_67 = data_flow_67.mean((0, 1))
print(data_flow.shape)

data_occ = data[864:14976, :, 1]
data_occ_1 = data_occ.reshape(-1, 7 * 288, 170).mean(0)
data_occ = data_occ.reshape(-1, 7, 288, 170)
data_occ_week = data_occ[:, :5, :, :]
data_occ_67 = data_occ[:, 5:, :, :]
data_occ_week = data_occ_week.mean((0, 1))  # (288,307)
data_occ_67 = data_occ_67.mean((0, 1))  # (288,307)

import matplotlib.pyplot as plt

data_speed = data[864:14976, :, 2]
data_speed_1 = data_speed.reshape(-1, 7 * 288, 170).mean(0)
data_speed = data_speed.reshape(-1, 7, 288, 170)
data_speed_week = data_speed[:, :5, :, :]
data_speed_67 = data_speed[:, 5:, :, :]
data_speed_week = data_speed_week.mean((0, 1))  # (288,307)
data_speed_67 = data_speed_67.mean((0, 1))  # (288,307)




import matplotlib.pyplot as plt
def plt_week(data,title): #(7*288,n)

    plt.plot(range(7 * 288), data, label='average')
    plt.legend()
    plt.title(title)
    plt.show()

# plt_week(data_flow_1.mean(1),'flow')
# plt_week(data_speed_1.mean(1),'speed')
# plt_week(data_occ_1.mean(1),'occ')
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
    # 计算每个相邻点之间的斜率

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

    max_slope_indices2 = np.argsort(np.abs(slopes12))[:100]
    max_slope_indices2.sort()
    max_slope_indices2 = max_slope_indices2[range(0,100,6)]

    max_slope_indices1.sort()

    index1=np.concatenate(([24],indices1,max_slope_indices1,max_slope_indices2,[311]),axis=-1)

    all_index=np.sort(np.unique(index1))



    total_area = cumulative_sum[-1]
    area_per_section = total_area / 50


    x_values = []
    current_area = 0
    for i in range(1, len(cumulative_sum)):
        if cumulative_sum[i] >= current_area + area_per_section:
            x_values.append(i)
            current_area += area_per_section
    x_values.sort()

    plt.plot(range(0,336), now_data,label='traffic')
    plt.scatter(max_slope_indices1, now_data[max_slope_indices1], color='red', label='largest 1st derivative')
    plt.scatter(max_slope_indices2, now_data[max_slope_indices2], color='green', label='smallest 1st derivative')

    plt.scatter(indices1, now_data[indices1], color='blue', label='peak')

    for t in index1:
       plt.axvline(x=t, color='c', linestyle='--')
    for t in [24,312]:
       plt.axvline(x=t, color='black', linestyle='--')

    plt.xlabel('time')
    plt.ylabel('flow')
    plt.legend()
    plt.title(title1)

    plt.show()

    return index1

def plt_smooth(data):
    x=range(0,288)


    #Savitzky-Golay滤波器
    from scipy.signal import savgol_filter
    window_length = 25 # 窗口长度
    poly_order = 1  # 多项式拟合阶数
    mean_data = savgol_filter(data, window_length, poly_order)


    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    model = SimpleExpSmoothing(data)
    fit = model.fit(smoothing_level=0.2)
    trend = fit.level


    plt.plot(x, data, label='1')
    plt.plot(x, mean_data, label='average')


    plt.xlabel('time')
    plt.ylabel('flow')
    plt.legend()
    plt.title("average  31")
    # 显示图形
    plt.show()
    return mean_data

#plt_smooth(data_flow_week.mean(-1))
flow_week=plt_day(data_flow_week.mean(-1),"flow week",window_length = 25,poly_order = 1,epsilon = 2)

#plt_smooth(data_flow_67.mean(-1))
flow_67=plt_day(data_flow_67.mean(-1),"flow 67",window_length = 23,poly_order = 1,epsilon = 2)


#plt_smooth(data_occ_week.mean(-1))
occ_week=plt_day(data_occ_week.mean(-1),"occ week",window_length = 31,poly_order = 2,epsilon = 0.001)

occ_67=plt_day(data_occ_67.mean(-1),"occ 67",window_length = 23,poly_order = 1,epsilon = 0.001)

speed_week=plt_day(data_speed_week.mean(-1),"speed week",window_length = 29,poly_order = 1,epsilon = 0.5)

speed_67=plt_day(data_speed_67.mean(-1),"speed 67",window_length = 25,poly_order = 1,epsilon = 0.01)


week_set = set(flow_week) | set(occ_week) | set(speed_week)
week_list = list(week_set)
week_list.sort()
print(week_list)
print(len(flow_week),len(occ_week),len(speed_week),len(week_list))
week_list=np.array(week_list)

week_list=week_list[week_list>23]
week_list=week_list[week_list<312]
week_list=week_list-24

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

def plot_all(data,point,title,window_length, poly_order):
    from scipy.signal import savgol_filter

    data = savgol_filter(data, window_length, poly_order)

    plt.plot(range(0,288), data,label='1')
    plt.scatter(point, data[point], color='red', label='Max Slope Points')
    for t in point:
       plt.axvline(x=t, color='c', linestyle='--')
    plt.legend()
    plt.title(title)
    plt.show()

plot_all(data_flow_week.mean(-1),week_list1,'flow',window_length=29, poly_order=3)
plot_all(data_occ_week.mean(-1),week_list1,'occ',window_length=27, poly_order=3)
plot_all(data_speed_week.mean(-1),week_list1,'speed',window_length=27, poly_order=3)

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


week=[0, 6, 15, 18, 26, 29, 36, 39, 42, 48, 54, 57, 60, 66, 72, 75, 78, 83, 86, 89, 92,
      95, 98, 101, 104, 107, 112, 118, 121, 159, 162, 165, 169, 177, 182, 189, 192, 195,
      199, 205, 210, 213, 216, 219, 222, 225, 228, 231, 234, 237, 243, 249, 252, 255, 260,
      264, 267, 270, 273, 276, 279, 282, 285]
list67=[0, 6, 12, 15, 18, 21, 27, 31, 34, 41, 46, 50, 54, 57, 66, 69, 72, 75, 78, 81, 84,
        87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126, 129, 134, 140, 144,
        148, 152, 155, 159, 163, 166, 171, 178, 184, 190, 196, 202, 208, 214, 220, 223, 229,
        233, 236, 239, 243, 246, 249, 255, 260, 264, 267, 270, 274, 280, 283, 286]



