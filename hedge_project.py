# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 15:21:13 2021

@author: Owner
"""
import sys
sys.path.append(r'C:\Users\Owner\Documents\University\PhD\Code\Python\AA_coursework.py')
from AA_coursework import AA, squared_loss, absoloute_loss, AA_Class, weak_AA_class
#import importlib.util
#spec = importlib.util.spec_from_file_location("add", "C:\\Users\\Shubham-PC\\PycharmProjects\\pythonProject1")
import matplotlib.pyplot as plt



import numpy as np
import  pandas as pd
import itertools

from LossFunction import *

def hedge_experts(limits, hedge_fractions):
    preds = []
    for p in itertools.product(limits, hedge_fractions,  repeat = 2):
        preds.append(p)
    preds_arr = np.array(preds)
    preds_arr[:,0] = preds_arr[:,0]*1000000
    preds_arr[:,2] = preds_arr[:,2]*-1000000
    return preds_arr
  
def hedge_experts_skew(limits, hedge_fractions, skew,window):
    preds = []
    for p in itertools.product(limits, hedge_fractions, skew, window, repeat = 1):
        preds.append(p)
    preds_arr = np.array(preds)
    preds_arr[:,0] = preds_arr[:,0]*1000000
    return preds_arr

def hedge_fraction_prediction(hedge_experts, NetPos):
    T = len(NetPos)
    N_experts = np.size(hedge_experts,0)
    prediction_array = np.zeros(( T , N_experts ))
    for t in range(T):
        for i in range(N_experts):
            if  hedge_experts[i,0] < NetPos[t]:
                prediction_array[t,i] = hedge_experts[i,1]
            elif hedge_experts[i,2] > NetPos[t]:             
                prediction_array[t,i] = hedge_experts[i,3]
    return prediction_array
   
def outcome_pnl(pnl, NetPos, absvol, mid_price):
    T = len(pnl)
    outcome = np.ones(T)
    for t in range(T-1):
        outcome[t] = pnl[t+1] / ( abs(NetPos[t]) + absvol[t+1] ) #(mid_price[t+1] - mid_price[t]) / mid_price[t] #-pnl[t+1] / ( abs(NetPos[t]) + absvol[t+1] )
        
    return outcome

def outcome_binary(pnl):
    T = len(pnl)
    outcome = np.ones(T)
    for t in range(T):
        if pnl[t] < 0:
            outcome[t] = 1
        else:
            outcome[t] = 0
    return outcome
    
def long_short_loss(outcome, prediction):
    N_assets = 1
    return_to_portfolio = 0
    for n in range(N_assets):
        return_to_portfolio += outcome * prediction
    loss = -np.log(1 + return_to_portfolio)
    return loss
    
def combined_loss(outcome, prediction, return_scale = 400, ls = 1, dls = 0):
    N_assets = 1
    return_to_portfolio = 0
    for n in range(N_assets):
        return_to_portfolio +=  ((ls / (ls + dls)) * return_scale*(outcome * prediction)) + ( (dls / (ls + dls)) *return_scale* min(outcome * prediction, 0))
    loss = -np.log(1 + return_to_portfolio)
    return loss

def PnLs(pnl, hedge):
    client = pnl
    hedge = -pnl * hedge
    root = client + hedge
    return client, hedge, root

def drawdown(pnl):
    return pnl.cumsum() - pnl.cumsum().cummax()

def root_drawdown(preds, pnl):
    N = len(prediction_array[0,:])
    T = len(prediction_array[:,0])
    root_drawdowns = np.zeros((T, N))
    root__max_drawdowns = np.zeros((T, N))
    
    for n in range(N):
        root_pnl = (preds * pnl)
        root_pnl = root_pnl.cumsum()
        max_DD = 0
        for t in range(T):
            DD = drawdown(root_pnl[t])
            max_DD = min(DD, max_DD)
            root_drawdowns[t, n] = DD
            root__max_drawdowns[t, n] = max_DD
    return root_drawdowns, root__max_drawdowns
    
def hedge_pnls(hedge_predictions, client_pnl):
    N_models = len(hedge_predictions[0])
    trials = len(hedge_predictions[:,0])
    hedge_pnl = np.zeros((trials, N_models))
    for n in range(N_models):
        hedge_pnl[:,n] = hedge_predictions[:, n] * client_pnl
    return hedge_pnl

def moving_average(data, window):
    data = pd.Series(data)
    return data.rolling(window).mean()

def indicator(data, window):
    T = len(data)
    Average = moving_average(data, abs(window))

    skew_indicator = np.zeros(T)
    
    for t in range(T):
        if(window < 0):
            if Average[t] > data[t]:
                skew_indicator[t] = 1
            elif Average[t] < data[t]:
                skew_indicator[t] = -1
        else:      
            if Average[t] > data[t]:
                skew_indicator[t] = -1
            elif Average[t] < data[t]:
                skew_indicator[t] = 1
    return np.roll(skew_indicator,1) 


    
    
def skew_preds_hedge_fraction(hedge_experts, NetPos, price, window):
    Average = moving_average(price, window)
    price = price.values
    Average = Average.values
    T = len(NetPos)
    N_experts = np.size(hedge_experts,0)
    prediction_array = np.zeros(( T , N_experts ))
    

        
    for t in range(T):
        skew_indicator = 0
        if Average[t] > price[t]:
            skew_indicator = -1
        elif Average[t] < price[t]:
            skew_indicator = 1
        
        for i in range(N_experts):
            if  (hedge_experts[i,0] * hedge_experts[i,2] * skew_indicator) < NetPos[t]:
                prediction_array[t,i] = hedge_experts[i,1]
            elif (hedge_experts[i,0] * hedge_experts[i,2] * skew_indicator * -1) > NetPos[t]:             
                prediction_array[t,i] = hedge_experts[i,1]
    return prediction_array


def grid_agents(agents, netpos):
    results = []
    for x,agent in enumerate(agents):
        print("{0}/{1}".format(x,len(agents)))
        hedges = []
        for index,nop in enumerate(netpos) :
            hedges.append(agent.Action(nop,index)) 
        results.append(hedges)
    
    return np.array(results).T
    
class Agent(object):
    def __init__(self,indicator,limits, hedge_fractions, skew,window):
        self.indicator = indicator
        self.limits = limits
        self.hedge_fractions = hedge_fractions
        self.skew = skew
        self.window = window
        
    def Action(self,nop: float,index:int):
        if self.limits + (self.limits * self.skew *self.indicator[index]) < nop:
            return self.hedge_fractions
        elif  -self.limits + (self.limits * self.skew *self.indicator[index]) > nop:
            return self.hedge_fractions
        
        else:
            return 0
        
    def __str__(self):
        return "limits:{},hedge_fractions{},skew{},window{}".format(self.limits,self.hedge_fractions,self.skew,self.window )
        
import multiprocessing as mp

def find_top_max(hedge_pnls,nth):
    tt = hedge_pnls.cumsum(axis = 0).T
    col = tt.columns[-1]
    t1 = tt.sort_values(by=[col],ascending=False)  
    return t1.head(nth)     

def top_experts(predictions, hedge_pnls, nth):
    sort = find_top_max(hedge_pnls,nth)
    pred_sub = pd.DataFrame(predictions)
    ps = pred_sub.T.loc[sort.index[:]].T.values
    return ps

"""
    
    data['QdfTime'] = pd.to_datetime(data['QdfTime'])
    data_eur_usd = data[data['Symbol'] == 'EUR/USD']
    data_eur_usd['mid_price'] = data_eur_usd['Mid']
    data_datesum = data_eur_usd.groupby(['QdfTime']).sum()
    data_datesum = data_datesum.loc['2015-02-01':'2016-04-27']
    
    
    limits = (5, 10, 15, 20, 25, 30, 50, 100)
    hedge_fractions = (0.1, 0.25, 0.5, 0.75, 0.9)
    skew = (0.1,0.3,0.6,0.8)
    window = (-24*60,-24*30,-24*14,-24*7,-24,-12,-3,3,12, 24, 24*7, 24*14,24*30,24*60)
    
    indicators = {}
    for w in window:
        indicators[w] = indicator(data_datesum['mid_price'].values,w)
    
    
    
    hedge_array =  hedge_experts_skew(limits, hedge_fractions, skew,window)
    
    agents = []
    for s in hedge_array:
        agents.append(Agent(limits=s[0],hedge_fractions=s[1],skew = s[2],indicator=indicators[s[3]],window= s[3]))
    
    
    prediction_array = grid_agents(agents,data_datesum['NetPosUsd'].values)
    hedge_pnls = np.zeros((len(prediction_array),len(prediction_array[1,:])))
    for n in range(len(prediction_array[0,:])):
        hedge_pnls[:,n] = (prediction_array[:,n] * -data_datesum['NetUsdPnL'].values)
    hedge_pnls = pd.DataFrame(hedge_pnls)
    prediction_array = top_experts(prediction_array, hedge_pnls, 100)
    
    
    #prediction_array = skew_preds_hedge_fraction(hedge_array, data_datesum['NetPosUsd'].values,data_datesum['mid_price'], 24*7)
    
    #hedge_array =  hedge_experts(limits, hedge_fractions)
    #prediction_array = hedge_fraction_prediction(hedge_array, data_datesum['NetPosUsd'].values)
"""
if __name__ == '__main__':    
    agents = pd.read_pickle(r"C:\Users\Owner\Documents\University\PhD\Data\2021_data\preds_for_AA\agents.obj")
    data_datesum = pd.read_pickle(r"C:\Users\Owner\Documents\University\PhD\Data\2021_data\preds_for_AA\data.obj")
    #data_datesum = data_datesum.loc['2014-02-05':'2017-04-20']
    prediction_array = pd.read_pickle( r"C:\Users\Owner\Documents\University\PhD\Data\2021_data\preds_for_AA\predictions.obj")
    i = [12743,12855,13191,13079,12967,12631,12519,13303,13415,
         13415,
        13431,
        12309,
        12310,
        12311,
        13303,
        12855,
        12967,
        13303,
        13415,
        12407,
        13079,
        13191,
        12631,
        12743,
        12519,
        11495,
        11607,
        11831,
        12167,
        11943,
        11271,
        4424,
        5544,
        10047,
        4408,
        6655,
        5560,
        4312,
        5432,
        4200,
        9935,
        5320,
        4296,
        6543,
        5448,
        7863,
        8871,
        7975,
        8423,
        8199,
        8535,
        8087,
        8759,
        8647,
        8311,
        7970,
        8530,
        8194,
        8754,
        7858,
        8306,
        8642,
        8866,
        8418,
        8871,
        8759,
        8870,
        8865,
        8866,
        6689,
        8647,
        6694,
        8887,
        8758,
        8753,
        8754,
        6577,
        7735,
        7751,
        8535,
        8646,
        8641,
        12368,
        12592,
        12928,
        13376,
        13040,
        13264,
        12816,
        13152,
        12704,
        12480,
        11520,
        11296,
        11744,
        11968,
        12080,
        12192,
        11856,
        11408,
        11632,
        12304,
        12192,
        12080,
        11968,
        12288,
        13424,
        11184,
        12176,
        13312,
        11072,
        11856,
        13376,
        12064,
        13200,
        10960,
        13264,
        11952,
        13088,
        10848]
    prediction_array = prediction_array[:, i] * -1 
    #outcomes = outcome_pnl(data_datesum['NetUsdPnL'].values, data_datesum['NetPosUsd'].values, data_datesum['AbsVolume'].values, data_datesum['mid_price'].values)
    outcomes = data_datesum['NetUsdPnL'].values
    #outcomes_binary = outcome_binary(data_datesum['NetUsdPnL'].values)
    
    #outcomes = outcomes[9572:10984]
    #prediction_array = prediction_array[9572:10984, :]
    
#    cmblist = []
 #   for x in range(1,1000,10):
#        for y in range(1,2):
#            for z in range(0,1):
#                cmblist.append( CombinedLoss(return_scale = x, ls = y, dls = z))

    #Xlearner_loss, Xexpert_loss, Xlearner_preds = weak_AA_class(outcomes, prediction_array,cmb,1)
   
    
    # Step 1: Init multiprocessing.Pool()
#    pool = mp.Pool(mp.cpu_count())
    
    # Step 2: `pool.apply` the `howmany_within_range()`
#    results = [pool.apply(AA_Class, args=(outcomes, prediction_array,cmbdata,1)) for cmbdata in cmblist]
    
#    pool.close()
    
    #cmb = CombinedLoss(return_scale = 1, ls = 1, dls = 0)
    #cmb = PnLLoss(return_scale = 1, cof = 0.0001)
    #learner_loss, expert_loss, learner_preds = AA_Class(outcomes, prediction_array,cmb,1)
    
    cmb = PnL_weak_loss(return_scale = 1)
    
    Xlearner_loss, Xexpert_loss, Xlearner_preds  = weak_AA_class(outcomes, prediction_array,cmb,300000)
   # wlearner_loss, wexpert_loss, wlearner_preds = AA(outcomes_binary, prediction_array, absoloute_loss,2)

    plt.figure()
    #(data_datesum['NetUsdPnL'].cumsum() - data_datesum['NetUsdPnL'].cumsum().cummax() ).plot()
    x2 = Xlearner_preds * data_datesum['NetUsdPnL'].values
   # y = x + data_datesum['NetUsdPnL'].values
    plt.plot(x2.cumsum())
    #x = Xlearner_preds * data_datesum['NetUsdPnL'].values    #x[1] = 0
    #plt.plot(x.cumsum())
   # yweak = x + data_datesum['NetUsdPnL'].values
    #lt.figure()
    #data_datesum['NetUsdPnL'].cumsum() - data_datesum['NetUsdPnL'].cumsum().cummax() ).plot()
    #(pd.DataFrame(y, index = data_datesum['NetUsdPnL'].index).cumsum() - pd.DataFrame(y, index = data_datesum['NetUsdPnL'].index).cumsum().cummax() ).plot()
    #(pd.DataFrame(yweak, index = data_datesum['NetUsdPnL'].index).cumsum() - pd.DataFrame(yweak, index = data_datesum['NetUsdPnL'].index).cumsum().cummax() ).plot()
    
    
    #ax = data_datesum['NetUsdPnL'].cumsum().plot(fontsize = 20)
    #test2 = pd.DataFrame(x2, index = data_datesum['NetUsdPnL'].index)
    #test2.cumsum().plot(ax=ax)
    #test2['net'] = test2[0].values + data_datesum['NetUsdPnL'].values
    #test2['net'].cumsum().plot(ax=ax)
    #plt.xlabel('Time', fontsize = 40)
    #plt.ylabel('PnL', fontsize = 40)
    #plt.legend(['Client','Hedge', 'Net'])
    
    
    