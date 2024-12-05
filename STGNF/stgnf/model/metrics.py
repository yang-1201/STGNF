"""
Copyright 2022 ServiceNow
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

>> The methods to compute the metrics on the GluonTS forecast objects.
"""


import os
import pickle
import sys
from typing import Dict, Iterable, Iterator, Optional

import numpy as np
import pandas as pd
from gluonts import transform
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.evaluation import MultivariateEvaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.forecast import Forecast
from gluonts.model.predictor import Predictor
import properscoring as ps
import torch
from datetime import datetime

class SplitValidationTransform(transform.FlatMapTransformation):
    """
    Split a dataset to do validation tests ending at each possible time step.
    A time step is possible if the resulting series is at least as long as the window_length parameter.
    """

    def __init__(self, window_length: int):
        super().__init__()
        self.window_length = window_length

    def flatmap_transform(self, data: DataEntry, is_train: bool) -> Iterator[DataEntry]:
        full_length = data["target"].shape[-1]
        for end_point in range(self.window_length, full_length+1):
            data_copy = data.copy()
            #print(data_copy)
            data_copy["target"] = data["target"][..., :end_point]
            data_copy["feat_dynamic_real"] = data["feat_dynamic_real"][..., :end_point]
            data_copy["feat_dynamic_cat"] = data["feat_dynamic_cat"][..., :end_point]
            #print(data_copy)
            yield data_copy


class SuppressOutput:
    """
    Context controller to remove any printing to standard output and standard error.
    Inspired from:
    https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    """

    def __enter__(self):
        self._stdout_bkp = sys.stdout
        self._stderr_bkp = sys.stderr
        sys.stdout = sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._stdout_bkp
        sys.stderr = self._stderr_bkp


def _compute_energy_score(target_data: np.array, samples: np.array, num_samples: int, beta: float) -> np.float32:
    """
    Compute the unnormalized energy score for a single forecast.

    Parameters:
    -----------
    target_data: np.array [two dimensions]
        The ground truth.
    samples: np.array [number of samples, two additional dimensions]
        The samples from the forecasting method to be assessed.
    num_samples: int
        The number of samples from the forecast.
    beta: float
        The beta parameter for the energy score.

    Returns
    -------
    score: np.float32
        The energy score.
    """

    norm = np.linalg.norm(samples - target_data[None, :, :], ord="fro", axis=(1, 2))  #(s)

    first_term = (norm**beta).mean()

    # For the second term of the energy score, we need two independant realizations of the distributions.
    # So we do a sum ignoring the i == j terms (which would be zero anyway), and we normalize by the
    # number of pairs of samples we have summed over.
    s = np.float32(0)
    for i in range(num_samples - 1):
        for j in range(i + 1, num_samples):
            norm = np.linalg.norm(samples[i] - samples[j], ord="fro")  #1
            s += norm**beta


    second_term = s / (num_samples * (num_samples - 1) / 2)

    return first_term - 0.5 * second_term

def _compute_crps(target_data: np.array, samples: np.array, num_samples: int, beta: float) -> np.float32:
    """
    Compute the unnormalized energy score for a single forecast.

    Parameters:
    -----------
    target_data: np.array [two dimensions]
        The ground truth.
    samples: np.array [number of samples, two additional dimensions]
        The samples from the forecasting method to be assessed.
    num_samples: int
        The number of samples from the forecast.
    beta: float
        The beta parameter for the energy score.

    Returns
    -------
    score: np.float32
        The energy score.
    """
    # The Frobenius norm of a matrix is equal to the Euclidean norm of its element:
    # the square root of the sum of the square of its elements


    print(target_data.shape)  #(t,n)
    print(samples.shape)  #(s,t,n)
    norm = np.abs(samples - target_data)  #(s)
    first_term = norm.mean()

    # For the second term of the energy score, we need two independant realizations of the distributions.
    # So we do a sum ignoring the i == j terms (which would be zero anyway), and we normalize by the
    # number of pairs of samples we have summed over.
    s = np.float32(0)
    for i in range(num_samples - 1):
        for j in range(i + 1, num_samples):
            norm = np.abs(samples[i] - samples[j])  #1
            s += norm
    second_term = s / (num_samples * (num_samples - 1) / 2)  #1

    return first_term - 0.5 * second_term  #1




def compute_energy_score(
    targets: Iterable[pd.DataFrame], forecasts: Iterable[Forecast], beta: float = 1.0
) -> np.float32:
    """
    Compute the non-normalized energy score for a multivariate stochastic prediction from samples.

    Parameters:
    -----------
    targets: Iterable[pd.DataFrame]
        The observed values, containing both the history and the prediction windows.
        Each element is taken independantly, and the result averaged over them.
    dataset: Iterable[Forecast]
        An object containing multiple samples of the probabilistic forecasts.
        This iterable should have the exact same length as targets.
    beta: float, default to 1.
        The energy score parameter, must be between 0 and 2, exclusive.

    Returns:
    --------
    result: np.float32
        A dictionary containing the various metrics
    """
    assert 0 < beta < 2

    cumulative_score = np.float32(0)
    num_forecasts = 0
    for target, forecast in zip(targets, forecasts):  #每个数据
        # The targets should always end with the prediction window
        assert target.index[-forecast.prediction_length] == forecast.start_date

        target_data = target.iloc[-forecast.prediction_length :].to_numpy()
        samples = forecast.samples


        cumulative_score += _compute_energy_score(target_data, samples, forecast.num_samples, beta)

        num_forecasts += 1

    return cumulative_score / num_forecasts


def compute_mask_energy_score(
    targets: Iterable[pd.DataFrame], forecasts: Iterable[Forecast], beta: float = 1.0, test:torch.Tensor=None,
) -> np.float32:
    """
    Compute the non-normalized energy score for a multivariate stochastic prediction from samples.

    Parameters:
    -----------
    targets: Iterable[pd.DataFrame]
        The observed values, containing both the history and the prediction windows.
        Each element is taken independantly, and the result averaged over them.
    dataset: Iterable[Forecast]
        An object containing multiple samples of the probabilistic forecasts.
        This iterable should have the exact same length as targets.
    beta: float, default to 1.
        The energy score parameter, must be between 0 and 2, exclusive.

    Returns:
    --------
    result: np.float32
        A dictionary containing the various metrics
    """
    assert 0 < beta < 2

    cumulative_score = np.float32(0)
    num_forecasts = 0

    test=test.transpose(0,1)  #(s,t)->(t,s)
    test=torch.where(test>0,torch.zeros_like(test),torch.ones_like(test))

    history_length=36
    prediction_length=12
    for target, forecast in zip(targets, forecasts):  #每个数据
        # The targets should always end with the prediction window

        target_data = target.iloc[-(history_length+prediction_length) :].to_numpy()
        samples = forecast.samples

        target_mask_data=np.zeros_like(target_data)
        samples_mask=np.zeros_like(samples)

        test_mask=test[num_forecasts:num_forecasts+history_length]  #(t1,n)

        target_mask_data[:history_length,:]=test_mask*target_data[:history_length,:]
        target_mask_data[history_length:]=target_data[history_length:,:]
        samples_mask[:,:history_length,:]=test_mask[None,:,:]*samples[:,:history_length,:]
        samples_mask[:,history_length:,:]=samples[:,history_length:,:]

        cumulative_score += _compute_energy_score(target_mask_data, samples_mask, forecast.num_samples, beta)

        num_forecasts += 1
    return cumulative_score / num_forecasts



def compute_crps(
    targets: Iterable[pd.DataFrame], forecasts: Iterable[Forecast], beta: float = 1.0
) -> np.float32:
    """
    Compute the non-normalized energy score for a multivariate stochastic prediction from samples.

    Parameters:
    -----------
    targets: Iterable[pd.DataFrame]
        The observed values, containing both the history and the prediction windows.
        Each element is taken independantly, and the result averaged over them.
    dataset: Iterable[Forecast]
        An object containing multiple samples of the probabilistic forecasts.
        This iterable should have the exact same length as targets.
    beta: float, default to 1.
        The energy score parameter, must be between 0 and 2, exclusive.

    Returns:
    --------
    result: np.float32
        A dictionary containing the various metrics
    """
    assert 0 < beta < 2

    cumulative_score = np.float32(0)
    num_forecasts = 0
    for target, forecast in zip(targets, forecasts):   #196个预测的12时

        # The targets should always end with the prediction window
        assert target.index[-forecast.prediction_length] == forecast.start_date
        target_data = target.iloc[-forecast.prediction_length :].to_numpy()  #(t,n)
        samples = forecast.samples  #(s,t,n)
        num_samples=forecast.num_samples


        target_data=target_data.reshape(-1) #(tn)
        t_n=target_data.shape[0]
        samples=samples.reshape(num_samples,-1) #(s,tn)
        for i in range(t_n):
            #a=_compute_crps(target_data[i], samples[:,i], forecast.num_samples, beta)
            a=ps.crps_ensemble(target_data[i],samples[:,i])
            print(a)
            cumulative_score += a
            num_forecasts += 1
    return cumulative_score / num_forecasts

import torch
def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )
def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))
def calc_quantile_CRPS(target, forecast, eval_points):
    """
    target: (B, T, V), torch.Tensor
    forecast: (B, n_sample, T, V), torch.Tensor
    eval_points: (B, T, V): which values should be evaluated,
    """

    # target = target * scaler + mean_scaler
    # forecast = forecast * scaler + mean_scaler
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def compute_validation_metrics(
    predictor: Predictor,
    dataset: Dataset,
    window_length: int,
    num_samples: int,
    split: bool = True,
    savedir: Optional[str] = None,
    multi:int=0,
    predict_mask: torch.Tensor=None,
) -> Dict[str, float]:
    """
    Compute GluonTS metrics for the given predictor and dataset.

    Parameters:
    -----------
    predictor: Predictor
        The trained model to predict with.
    dataset: Dataset  N个{‘strat’:开始时间，‘target’:预测长度，item_id:"id"}  (20,3398)
        The dataset on which the model will be tested.
    window_length: int
        The prediction length + history length of the model.    prediction+history
    num_samples: int
        How many samples will be generated from the stochastic predictions.
    split: bool, default to True
        If set to False, the dataset is used as is, normally with one prediction per entry in the dataset.
        If set to True, the dataset is split into all possible subset, thus with one prediction per timestep in the dataset.
        Normally should be set to True during HP search, since the HP search validation dataset has only one entry;
        and set to False during backtesting, since the testing dataset has multiple entries.
    savedir: None or str, default to None
        If set, save the forecasts and the targets in a pickle file named forecasts_targets.pkl located in said location.

    Returns:
    --------
    result: Dict[str, float]
        A dictionary containing the various metrics.
    """
    #print(len(dataset))
    # for i  in dataset:
    #     print(i)
    #     print(i['target'].shape)  #(20,240)
    #原先是一个multigrouper{target:[n,t]}
    #现在是多个 n个 每个{target:[v,t]}
    if split:#(20,3398)-> (20,96)(20,97)(20,98)   (20,3397) 共3302个
        split_dataset = transform.TransformedDataset(dataset, transformation=SplitValidationTransform(window_length))
    else:
        split_dataset = dataset
    print("test data:"+str(len(split_dataset))) #(60,240)    n*分割的


    print("start:",datetime.now())
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=split_dataset, predictor=predictor, num_samples=num_samples
    )
    print("预测")

    forecasts = list(forecast_it)  #(b,sample,t2,v)
    targets = list(ts_it)
    print("end:", datetime.now())



    forecasts_list=[]
    targets_list=[]

    if savedir:
        savefile = os.path.join(savedir)
        with open(savefile, "wb") as f:
            pickle.dump((forecasts, targets), f)
            print("save")


    if predict_mask !=None:
        print("mask_Energy:", compute_mask_energy_score(targets, forecasts,test=predict_mask))




    for i in forecasts:#()

        forecasts_list.append(i.samples) #192个 (100,12,60)  #(b,sample,t2,v)

    for i in targets:

        targets_list.append(np.array(i)[-predictor.prediction_length:,:])#(48,60)




    forecasts_list=np.array(forecasts_list)  #(b,s,t,n)
    targets_list=np.array(targets_list)  #(b,t,nei*v)

    forecasts_list1 = forecasts_list  # (b,s,t,n)
    targets_list1 = targets_list  # (b,t,n)


    forecasts_list =forecasts_list.transpose(0, 2, 3, 1).mean(axis=-1) #(b,t,n,s)->(b,t,n)


    num_series=forecasts_list.shape[-1]
    targets_list=targets_list[:,:,:num_series] #(b,t,v)
    b,t,c=targets_list.shape


    print(forecasts_list.shape)
    if multi>1:
        node=c//multi
        for i in range(multi):
            index = [j * multi + i for j in range(node)]  #i=0 3 i=1

            forecasts_dim=forecasts_list[:,:,index]  #(b,t,n)
            targets_dim=targets_list[:,:,index]


            mae,rmse,mape,mse=All_Metrics(forecasts_dim, targets_dim)
            print("{}, MAE: {:.5f}, RMSE: {:.4f},MAPE:{:.4f}%  MSE: {:.3f}".format(i,
                    mae, rmse,mape*100,  mse))

    mae, rmse, mape, mse = All_Metrics(forecasts_list, targets_list)
    print("ALL MAE: {}, RMSE: {},MAPE:{}% MSE: {}".format(
             mae, rmse, mape*100, mse))

    eval_points = np.ones_like(targets_list1)
    print("crps:", calc_quantile_CRPS(torch.from_numpy(targets_list1), torch.from_numpy(forecasts_list1), eval_points))

    eval_points_sum = np.ones_like(targets_list1.sum(-1))
    print("crps:",
          calc_quantile_CRPS(torch.from_numpy(targets_list1.sum(-1)), torch.from_numpy(forecasts_list1.sum(-1)),
                             eval_points_sum))

    print("Energy:",compute_energy_score(targets, forecasts))

    # The results are going to be meaningless if any NaN shows up in the results,
    # so catch them here
    num_nan = 0
    num_inf = 0
    for forecast in forecasts:
        num_nan += np.isnan(forecast.samples).sum()
        num_inf += np.isinf(forecast.samples).sum()
    if num_nan > 0 or num_inf > 0:
        return {
            "CRPS": float("nan"),
            "ND": float("nan"),
            "NRMSE": float("nan"),
            "MSE": float("nan"),
            "CRPS-Sum": float("nan"),
            "ND-Sum": float("nan"),
            "NRMSE-Sum": float("nan"),
            "MSE-Sum": float("nan"),
            "Energy": float("nan"),
            "num_nan": num_nan,
            "num_inf": num_inf,
        }
    #from .TACTISEvaluator import MultivariateEvaluator1
    # Evaluate the quality of the model
    #evaluator = MultivariateEvaluator(quantiles=(np.arange(20) / 20.0)[1:], target_agg_funcs={"sum": np.sum})
    evaluator = MultivariateEvaluator(quantiles=(np.arange(20) / 20.0)[1:], target_agg_funcs={"sum": np.sum},num_workers=2)

    # The GluonTS evaluator is very noisy on the standard error, so suppress it.
    with SuppressOutput():
        agg_metric, item_metric = evaluator(targets, forecasts)
        #agg_metric, item_metric = evaluator(targets, forecasts,num_workers=2)

    #item_metric (192测试*n,c测试) 每个节点 测试时的数据 n1 192 n2 192..nn 192
    return {
        "CRPS": agg_metric.get("mean_wQuantileLoss", float("nan")),
        "ND": agg_metric.get("ND", float("nan")),
        "NRMSE": agg_metric.get("NRMSE", float("nan")),
        "MSE": agg_metric.get("MSE", float("nan")),
        "RMSE": agg_metric.get("RMSE", float("nan")),
        "mae":agg_metric.get("mae", float("nan")),
        "mae1": get_mae(agg_metric),
        "MAPE": agg_metric.get("MAPE", float("nan")),
        "MASE": agg_metric.get("MASE", float("nan")),
        "seasonal_error": agg_metric.get("seasonal_error", float("nan")),
        "CRPS-Sum": agg_metric.get("m_sum_mean_wQuantileLoss", float("nan")),
        "ND-Sum": agg_metric.get("m_sum_ND", float("nan")),
        "NRMSE-Sum": agg_metric.get("m_sum_NRMSE", float("nan")),
        "MSE-Sum": agg_metric.get("m_sum_MSE", float("nan")),
        "item_id":agg_metric.get("item_id", float("nan")),
        "1_MSE": agg_metric.get("1_MSE", float("nan")),
        "Energy": compute_energy_score(targets, forecasts),
        "num_nan": num_nan,
        "num_inf": num_inf,
    }


def get_mae(agg_metric):
    return agg_metric.get('MASE')*agg_metric.get("seasonal_error")


def RMSE_np(pred, true, mask_value=None):

    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    #print(np.sqrt(np.square(pred-true)))
    RMSE = np.sqrt(np.mean(np.square(pred-true)))
    return RMSE
def MSE_np(pred, true, mask_value=None):

    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    MSE = np.mean(np.square(pred-true))
    return MSE

def MAE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    MAE = np.mean(np.absolute(pred-true))
    return MAE

def MAPE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]

    mape = np.abs(np.divide(np.subtract(pred, true),
                      true))

    return np.mean(mape)


def All_Metrics(pred, true, mask1=None):
    assert type(pred) == type(true)
    if type(pred) == np.ndarray:
        mae = MAE_np(pred, true, mask1)
        rmse = RMSE_np(pred, true, mask1)
        mape = MAPE_np(pred, true, 0.)
        #mape=None
        mse=MSE_np(pred, true, mask1)
    else:
        raise TypeError
    return mae, rmse,  mape,mse


