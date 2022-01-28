import pandas   as pd
import pickle5 as pickle
from AA_coursework import AA, squared_loss, absoloute_loss, AA_Class, weak_AA_class
from LossFunction import *
import matplotlib.pyplot as plt
import numpy as np

# Peak to peak
def rolling_with_step_idx(window,step,length):
    samples_idx = []
    idx = 0
    while idx+window+step < length:
        samples_idx.append((idx, idx+window+step))
        idx += step

    return samples_idx


if __name__ == '__main__':
    with open("agents.obj", "rb") as fh:
        agents = pickle.load(fh)
    with open("data.obj", "rb") as fh:
        data_datesum = pickle.load(fh)
    with open("preds.obj", "rb") as fh:
        prediction_array = pickle.load(fh)

    batch_size = 24*60
    testing_size = 24*7



    outcomes = data_datesum['NetUsdPnL'].values
    idx = rolling_with_step_idx(batch_size,testing_size,outcomes.__len__())


    net_pnl = data_datesum['NetUsdPnL'].reset_index(drop=True)
    results = []
    process = 1
    for idx_range in idx:
        print(f"{process}/{len(idx)}")
        outcomes_s = outcomes[idx_range[0]:idx_range[1]]
        prediction_array_s = prediction_array[idx_range[0]:idx_range[1],:]
        cmb = PnL_weak_loss(return_scale=1)
        Xlearner_loss, Xexpert_loss, Xlearner_preds  = weak_AA_class(outcomes_s, prediction_array_s,cmb,300000)
        x2 = (Xlearner_preds * net_pnl[idx_range[0]:idx_range[1]].values)[idx_range[1]-testing_size:idx_range[1]]
        results.append(x2)
        process+=1
    # x2 = Xlearner_preds * data_datesum['NetUsdPnL'].values
    # y = x + data_datesum['NetUsdPnL'].values
    results_array = np.concatenate(results)
    plt.plot(results_array.cumsum())
    plt.show()