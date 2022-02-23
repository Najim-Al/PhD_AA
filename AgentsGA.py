import numpy as np
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
from hedge_project import *


import pickle5 as pickle
with open('data.obj', "rb") as fh:
  data_datesum = pickle.load(fh)

model_param = []


algorithm_param = {'max_num_iteration': 100,\
                   'population_size':100,\
                   'mutation_probability':0.2,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'function_timeout ':60,
                   'max_iteration_without_improv':None}

varbound=np.array([[0,1e8], #limits
                   [-1,0], #fractions
                   [0,1], #skew
                   [-1440,1440]  #window
                   ])

#varibal type
vartype=np.array([['int'],['real'],['real'],['int']])


def fitness_func_pnl(X):
    limit,fraction,skew,window = int(X[0]),X[1],X[2],int(X[3])
    ind = indicator(data_datesum['mid_price'].values, window)
    agent = Agent(limits=limit,hedge_fractions=fraction,skew = skew,indicator=ind,window= window)
    netpos = data_datesum['NetPosUsd'].values
    hedges = []
    for index, nop in enumerate(netpos):
        hedges.append(agent.Action(nop, index))

    results = np.array(hedges)
    pnl_result = (results)*data_datesum['NetUsdPnL']
    root_pnl = pnl_result+data_datesum['NetUsdPnL']

    # print(data_datesum['NetUsdPnL'].sum())

    hedge = pnl_result.sum()
    print(hedge)


    root = root_pnl.sum()
    # print(root)

    cumpnl = root_pnl.cumsum()
    meanDD = (np.maximum.accumulate(cumpnl) - cumpnl).mean()
    model_param.append({
        'PnL':root,
        'MeanDD':meanDD,
        'param': f'limits:{limit}, hedge_fractions:{fraction},skew:{skew},window:{window}'

    })
    print(f"'PnL':{root},'MaxDD':{meanDD},'limits"
          f":{limit}, hedge_fractions:{fraction},skew:{skew},window:{window}'")

    return -1.0 * hedge

def fitness_func(X):
    limit,fraction,skew,window = int(X[0]),X[1],X[2],int(X[3])
    ind = indicator(data_datesum['mid_price'].values, window)
    agent = Agent(limits=limit,hedge_fractions=fraction,skew = skew,indicator=ind,window= window)
    netpos = data_datesum['NetPosUsd'].values
    hedges = []
    for index, nop in enumerate(netpos):
        hedges.append(agent.Action(nop, index))

    results = np.array(hedges)
    pnl_result = results*data_datesum['NetUsdPnL'].values
    print((results*data_datesum['NetUsdPnL']).sum())

    pnl = pnl_result.sum()
    print(pnl)

    cumpnl = pnl_result.cumsum()
    maxDD = (np.maximum.accumulate(cumpnl) - cumpnl).max()
    model_param.append({
        'PnL':pnl,
        'MaxDD':maxDD,
        'param': f'limits:{limit}, hedge_fractions:{fraction},skew:{skew},window:{window}'

    })

    return -1.0 * pnl/maxDD if maxDD > 0 else 0

# res = fitness_func(np.array([20,-0.2,0.2,40]))
# print(res)

model=ga(function=fitness_func_pnl,dimension=4,variable_type_mixed=vartype,variable_boundaries=varbound,algorithm_parameters=algorithm_param)

model.run()
model_result = pd.DataFrame.from_dict(data=model_param)
model_result.to_csv('model_result.csv')

print(model.param)

