import numpy as np
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
from hedge_project import *


#import pickle5 as pickle
#with open('data.obj', "rb") as fh:
#  data_datesum = pickle.load(fh)

data_datesum = pd.read_pickle(r"C:\Users\Owner\Documents\University\PhD\Data\feb_21_paper\gbp_usd.obj")#read_pickle('data.obj')

model_param = []


algorithm_param = {'max_num_iteration': 200,\
                   'population_size':50,\
                   'mutation_probability':0.2,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'function_timeout ':60,
                   'max_iteration_without_improv':None}

# varbound=np.array([[0,1e8], #limits
#                    [-1,0], #fractions
#                    [0,1], #skew
#                    [-1440,1440]  #window
#                    ])


varbound=np.array([[0,1.5e8], #upper limits
                   [0,1.5e8], #lower limits
                   [-1,0], #upper fractions
                   [-1,0], #lower fractions
                   [0,1], #upper skew
                   [0,1], #lower skew
                   [-3000,3000]  #window
                   ])




#varibal type
vartype=np.array([['int'],['int'],['real'],['real'],
                  ['real'],['real'],['int']])


def fitness_func_pnl(X):
    upper_limit, lower_limit, upper_fraction,lower_fraction, upper_skew, lower_skew, window = int(X[0]),int(X[1]),X[2],X[3],X[4],X[5],int(X[6])
    ind = indicator(data_datesum['mid_price'].values, window)
    agent = Agent_new(upper_limit=upper_limit,lower_limit =lower_limit,
                      upper_hedge_fraction=upper_fraction, lower_hedge_fraction = lower_fraction,
                      upper_skew = upper_skew, lower_skew = lower_fraction, indicator=ind,window= window)
    netpos = data_datesum['NetPosUsd'].values
    hedges = []
    for index, nop in enumerate(netpos):
        hedges.append(agent.Action(nop, index))

    results = np.array(hedges)
    pnl_result = (results[0:-1])*data_datesum['NetUsdPnL'].values[1::]
    root_pnl = pnl_result+data_datesum['NetUsdPnL'].values[1::]

    # print(data_datesum['NetUsdPnL'].sum())

    hedge = pnl_result.sum()
    root = root_pnl.sum()
    # print(root)

    cumpnl = root_pnl.cumsum()
    meanDD = (np.maximum.accumulate(cumpnl) - cumpnl).mean()
    model_param.append({
        'PnL':root,
        'Root PnL': root_pnl,
        'MeanDD':meanDD,
        'Hedge fractions': results,
        'param': f'limits:{upper_limit, lower_limit}'
                 f'hedge_fractions:{upper_fraction, lower_fraction},'
                 f'skew:{upper_skew, lower_skew},window:{window}'

    })
    # print(f"'PnL':{root},'MaxDD':{meanDD},'limits"
    #       f":{limit}, hedge_fractions:{fraction},skew:{skew},window:{window}'")

    return -1.0 * root 

def fitness_func(X):
    limit,fraction,skew,window = int(X[0]),X[1],X[2],int(X[3])
    ind = indicator(data_datesum['mid_price'].values, window)
    agent = Agent(limits=limit,hedge_fractions=fraction,skew = skew,indicator=ind,window= window)
    netpos = data_datesum['NetPosUsd'].values
    hedges = []
    for index, nop in enumerate(netpos):
        hedges.append(agent.Action(nop, index))

    results = np.array(hedges)
    hedge_pnl = results*data_datesum['NetUsdPnL'].values
    root_pnl = hedge_pnl + data_datesum['NetUsdPnL'].values
    pnl = root_pnl.sum()
    print(pnl)

    cumpnl = root_pnl.cumsum()
    maxDD = (np.maximum.accumulate(cumpnl) - cumpnl).max()
    meanDD = (np.maximum.accumulate(cumpnl) - cumpnl).mean()
    print(meanDD)
    model_param.append({
        'PnL':pnl,
        'Root PnL':root_pnl,
        'Hedge PnL': hedge_pnl,
        'Mean DD': meanDD,
        'MaxDD':maxDD,
        'param': f'limits:{limit}, hedge_fractions:{fraction},skew:{skew},window:{window}'

    })

    return -1.0 * pnl/maxDD if maxDD > 0 else 0

# res = fitness_func(np.array([20,-0.2,0.2,40]))
# print(res)

model=ga(function=fitness_func_pnl,dimension=7,variable_type_mixed=vartype,variable_boundaries=varbound,algorithm_parameters=algorithm_param)

model.run()
model_result = pd.DataFrame.from_dict(data=model_param)
model_result.to_csv('model_result_gbp_usd_with_hedge.csv')

print(model.param)

