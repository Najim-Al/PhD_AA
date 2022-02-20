# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 15:21:13 2021

@author: Owner
"""
import sys
sys.path.append(r'C:\Users\Owner\Documents\University\PhD\Code\Python\AA_coursework.py')
from AA_coursework import *
#import importlib.util
#spec = importlib.util.spec_from_file_location("add", "C:\\Users\\Shubham-PC\\PycharmProjects\\pythonProject1")
import matplotlib.pyplot as plt
from tabulate import tabulate




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
        outcome[t] = pnl[t+1] / ( abs(NetPos[t]) + absvol[t+1] +1) #(mid_price[t+1] - mid_price[t]) / mid_price[t] #-pnl[t+1] / ( abs(NetPos[t]) + absvol[t+1] )
        
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
    N = len(preds[0,:])
    T = len(preds[:,0])
    root_drawdowns = np.zeros((T, N))
    root__max_drawdowns = np.zeros((T, N))
    
    for n in range(N):
        root_pnl = pd.DataFrame(pnl + (preds[:,n] * pnl))
        max_DD = 0
        for t in range(1,T-1):
            print('clinet:{}, epoch:{}'.format(n, t))
            DD = drawdown(root_pnl[:t]).iloc[-1].values
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

from random import randint
from math import floor

def pred_selector(prediction_array, PnL):
    t = np.size(prediction_array, axis = 0)
    intervals = np.linspace(0, t, 5)
    pnl = prediction_array*PnL[:,np.newaxis]
    prediction_array = pd.DataFrame(prediction_array)
    cumsum_pnls = np.cumsum(pnl, axis = 0)
    cumsum_pnls = pd.DataFrame(cumsum_pnls)
    column_index = []
    for n in range(4):
        print(intervals[n+1])
        interval_pnl = cumsum_pnls.iloc[floor(intervals[n+1]) -1 ]
        column_index.extend(interval_pnl.sort_values().iloc[0:10].index)
        column_index.extend(interval_pnl.sort_values().iloc[-10:].index)
        #column_index.extend(interval_pnl.iloc[6720].index)
    for _ in range(60):
        column_index.extend([randint(0,12320)])
    prediction_array = prediction_array.iloc[:,column_index]
    return prediction_array.values

def plot_DD(prediction_array, PnL, AA, AA_names):
    expert_pnl = prediction_array*PnL[:,np.newaxis]
    expert_pnl = expert_pnl + PnL[:,np.newaxis]
    expert_pnl = pd.DataFrame(expert_pnl)
    #AA_pnl = AA*PnL + PnL
    #AA_pnl = pd.DataFrame(AA_pnl)
    fig, ax = plt.subplots()
    ax.scatter(drawdown(expert_pnl).mean(),expert_pnl.cumsum().iloc[-1,:], color = 'blue')    
    for n in range(len(AA)):
        AA_pnl = AA[n] + PnL
        AA_pnl = pd.DataFrame(AA_pnl)
        ax.scatter(drawdown(AA_pnl).mean(),AA_pnl.cumsum().iloc[-1,:], color = 'red')
        ax.annotate(AA_names[n], (drawdown(AA_pnl).mean(),AA_pnl.cumsum().iloc[-1,:]))
    plt.xlabel('Mean Drawdown', fontsize = 40)
    plt.ylabel('PnL', fontsize = 40)
    
def wealth_prediction(hedge_fractions, client_pos, client_pnl, initial_wealth):
    N_clients = len(hedge_fractions[0,:])
    T = len(hedge_fractions[:,0])
    client_wealth = np.zeros((T+1,N_clients))
    client_wealth_t = np.zeros(N_clients)
    client_root = np.zeros((T,N_clients))
    client_wealth[0] += initial_wealth
    client_wealth_t[:] = initial_wealth
    predictions = np.zeros((T, N_clients))
    for n in range(N_clients):
        for t in range(T):
            client_root[t,n] = client_pos[t] * (1 + hedge_fractions[t,n])
            predictions[t,n] = client_root[t,n] / client_wealth[t,n]
            client_wealth_t[n] += client_pnl[t] * (1 + hedge_fractions[t,n])
            client_wealth[t+1,n] = client_wealth_t[n]
    return predictions,client_root,client_wealth
        
def learner_hedge_fraction(client_pos, client_wealth, learner_preds, weights):
        T = len(learner_preds)
        learner_hedge = np.zeros(T)
        for t in range(T):
            norm_weights = weights[t]/ sum(weights[t])
            norm_pos = norm_weights * client_pos[t]
            norm_pos = np.sum(norm_pos)
            norm_wealth = norm_weights * client_wealth[t]
            norm_wealth = np.sum(norm_wealth)
            learner_hedge[t] = (norm_pos / norm_wealth) - 1
        return learner_hedge
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
    agents = pd.read_pickle(r"agents.obj")
    data_datesum = pd.read_pickle(r"data.obj")
    prediction_array = pd.read_pickle(r"preds.obj")

    #data_datesum = data_datesum.loc['2014-02-05':'2017-04-20']
    #prediction_array = pd.read_pickle(r"C:\Users\Owner\Documents\University\PhD\initial hedge\paper\72expertpreds.obj")
    #DrawDown = pd.read_pickle(r"C:\Users\Owner\Documents\University\PhD\initial hedge\paper\72expertpredsdrawdown.obj")
    
    
    data_datesum = pd.read_pickle(r"C:\Users\Owner\Documents\University\PhD\Data\2021_data\preds_for_AA\data_chf_usd.obj")
    agents = pd.read_pickle( r"C:\Users\Owner\Documents\University\PhD\Data\2021_data\preds_for_AA\agents_chf_usd.obj")
    prediction_array = pd.read_pickle(r"C:\Users\Owner\Documents\University\PhD\initial hedge\paper\USD_CHF_PREDS.obj")
    prediction_array = pd.read_pickle(r"C:\Users\Owner\Documents\University\PhD\initial hedge\paper\CHF_USD_0_drop.obj")

 #   prediction_array_new,client_root,client_wealth = wealth_prediction(prediction_array, data_datesum['NetPosUsd'].values, data_datesum['NetUsdPnL'].values, 1*10**10 )

#    data_datesum = pd.read_pickle(r"C:\Users\Owner\Documents\University\PhD\Data\2021_data\preds_for_AA\data_eur_gbp.obj")
#   agents = pd.read_pickle(r"C:\Users\Owner\Documents\University\PhD\Data\2021_data\preds_for_AA\agents_eur_gbp.obj")
#    prediction_array = pd.read_pickle(r"C:\Users\Owner\Documents\University\PhD\initial hedge\paper\GBP_EUR_PREDS.obj")
    
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
    #prediction_array = prediction_array * -1 
    outcomes = outcome_pnl(data_datesum['NetUsdPnL'].values, data_datesum['NetPosUsd'].values, data_datesum['AbsVolume'].values, data_datesum['mid_price'].values)
    #outcomes = data_datesum['NetUsdPnL'].values
    #outcomes_binary = outcome_binary(data_datesum['NetUsdPnL'].values)
    #outcomes = outcomes[:1000]
    #prediction_array = prediction_array[:1000,:]
    #data_datesum = data_datesum.iloc[:1000]
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
    

# =============================================================================
    AA_hedge_pnls = []
    AA_root_pnls = []
    DD = []
    calmar = []
    
    list_dict = []
    
    # #run equal weights
    # cmb = CombinedLoss(return_scale = 1, ls = 1, dls = 0)
    # xlearner_loss, xexpert_loss, xlearner_preds,weight_L  = AA_Class(outcomes, prediction_array,cmb,1)
    # hedge_pnl_0 = xlearner_preds * data_datesum['NetUsdPnL'].values
    # AA_pnls.append(hedge_pnl_1)
    # print('Trial {} complete'.format(cmb))
    
    
    #Equal weights
    outcomes = outcome_pnl(data_datesum['NetUsdPnL'].values, data_datesum['NetPosUsd'].values, data_datesum['AbsVolume'].values, data_datesum['mid_price'].values)
    cmb = CombinedLoss(return_scale = 1, ls = 1, dls = 0)
    xlearner_loss, xexpert_loss, xlearner_preds, weights  = AA_Class(outcomes, prediction_array,cmb,1)
    AA_hedge_pnls = xlearner_preds * data_datesum['NetUsdPnL'].values
    #AA_pnls.append(hedge_pnl_1)
    AA_root_pnls = data_datesum['NetUsdPnL'].values + AA_hedge_pnls
    dataf = pd.DataFrame(AA_root_pnls)
    dd = drawdown(dataf)
    DD = dd.min()
    calmar = AA_root_pnls.sum() / abs(DD)
    AA = {'Name': 'Equal Weights',
          'Hedge PnL': AA_hedge_pnls,
          'Root PnL': AA_root_pnls,
          'Calmar': calmar}
    list_dict.append(AA)
    print('Trial {} complete'.format(cmb))
    
    # weak AA run
    outcomes = data_datesum['NetUsdPnL'].values
    cmb = PnL_weak_loss(return_scale = 1)
    xlearner_loss, xexpert_loss, xlearner_preds, weights  = weak_AA_class(outcomes, prediction_array,cmb,2000000)
    AA_hedge_pnls = xlearner_preds * data_datesum['NetUsdPnL'].values
    #AA_pnls.append(hedge_pnl_1)
    AA_root_pnls = data_datesum['NetUsdPnL'].values + AA_hedge_pnls
    dataf = pd.DataFrame(AA_root_pnls)
    dd = drawdown(dataf)
    DD = dd.min()
    calmar = AA_root_pnls.sum() / abs(DD)
    AA = {'Name': 'Weak AA',
          'Hedge PnL': AA_hedge_pnls,
          'Root PnL': AA_root_pnls,
          'Calmar': calmar}
    list_dict.append(AA)
    print('Trial {} complete'.format(cmb))
    
    
    #pnl AA games
    LS_values = [[1,0],
                 [1,1]]
    Discounting_values = [1, 0.05]

    for ls in range(len(LS_values)):
        for dis_factin in range(len(Discounting_values)):
            name = 'PnL loss, ls = {}, dls = {}, df = {}'.format(LS_values[ls][0],LS_values[ls][1], Discounting_values[dis_factin])
            cmb = LS_pnl_loss(return_scale = 0.000001, ls = LS_values[ls][0],
                               dls = LS_values[ls][1])
            xlearner_loss, xexpert_loss, xlearner_preds,weight_L  = AA_Class_discounted(outcomes, prediction_array,cmb,1,Discounting_values[dis_factin])
            AA_hedge_pnls = xlearner_preds * data_datesum['NetUsdPnL'].values
            AA_root_pnls = data_datesum['NetUsdPnL'].values + AA_hedge_pnls
            dataf = pd.DataFrame(AA_root_pnls)
            dd = drawdown(dataf)
            DD = dd.min()
            calmar = AA_root_pnls.sum() / abs(DD)
            AA = {'Name': name,
                  'Hedge PnL': AA_hedge_pnls,
                  'Root PnL': AA_root_pnls,
                  'Calmar': calmar}
            list_dict.append(AA)
            print('Trial {} complete'.format(cmb))
            

    
    #Standard AA games
    return_scaling_values = [1, 50, 100]
    LS_values = [[1,0],
                 [1,1]]
    Discounting_values = [1, 0.05]
    
    outcomes = outcome_pnl(data_datesum['NetUsdPnL'].values, data_datesum['NetPosUsd'].values, data_datesum['AbsVolume'].values, data_datesum['mid_price'].values)
    for ret in range(len(return_scaling_values)):
        for ls in range(len(LS_values)):
            for dis_factin in range(len(Discounting_values)):
                   
                name = 'return_scaling = {}, ls = {}, dls = {}, df = {}'.format( return_scaling_values[ret],
                                                                       LS_values[ls][0],
                                                                       LS_values[ls][1],
                                                                       Discounting_values[dis_factin])
                cmb = CombinedLoss(return_scale = return_scaling_values[ret], ls = LS_values[ls][0],
                                   dls = LS_values[ls][1])
                xlearner_loss, xexpert_loss, xlearner_preds,weight_L  = AA_Class_discounted(outcomes, prediction_array,cmb,1,Discounting_values[dis_factin])
                AA_hedge_pnls = xlearner_preds * data_datesum['NetUsdPnL'].values
                AA_root_pnls = data_datesum['NetUsdPnL'].values + AA_hedge_pnls
                dataf = pd.DataFrame(AA_root_pnls)
                dd = drawdown(dataf)
                DD = dd.min()
                calmar = AA_root_pnls.sum() / abs(DD)
                AA = {'Name': name,
                      'Hedge PnL': AA_hedge_pnls,
                      'Root PnL': AA_root_pnls,
                      'Calmar': calmar}
                list_dict.append(AA)
                print('Trial {} complete'.format(cmb))
    
    df = pd.DataFrame(list_dict)
    
    plt.figure()
    for n in range(len(df)):
        plt.plot(np.cumsum(df['Root PnL'].iloc[n]))
    plt.legend(df['Name'])
    
    plot_DD(prediction_array, data_datesum['NetUsdPnL'].values, df['Hedge PnL'], df['Name'])
        
    
    

    # for n in range(len(AA_pnls)):
    #         AA_root_pnls.append(sum(data_datesum['NetUsdPnL'].values + AA_pnls[n]))
    #         dataf = pd.DataFrame(AA_root_pnls[n])
    #         dd = drawdown(dataf)
    #         DD.append(dd.min())
    #         calmar.append(AA_root_pnls[n] / abs(DD[n]))












#     cmb = CombinedLoss(return_scale = 1, ls = 1, dls = 0)
#     xlearner_loss, xexpert_loss, xlearner_preds,weight_L  = AA_Class(outcomes, prediction_array,cmb,1)
#     x0 = xlearner_preds * data_datesum['NetUsdPnL'].values
#     AA_list.append(x0)
#     print('Trial {} complete'.format(cmb))
    
    
#     cmb = CombinedLoss(return_scale = 1, ls = 1, dls = 0)
#     xlearner_loss, xexpert_loss, xlearner_preds,weight_L  = AA_Class_discounted(outcomes, prediction_array,cmb,1,1)
#     x1 = xlearner_preds * data_datesum['NetUsdPnL'].values
#     AA_list.append(x1)
#     print('Trial {} complete'.format(cmb))
    
#     cmb = CombinedLoss(return_scale = 50, ls = 1, dls = 0)
#     xlearner_loss, xexpert_loss, xlearner_preds,weight_L  = AA_Class_discounted(outcomes, prediction_array,cmb,1,1)
#     x2 = xlearner_preds * data_datesum['NetUsdPnL'].values
#     AA_list.append(x2)
#     print('Trial {} complete'.format(cmb))

#     cmb = CombinedLoss(return_scale = 100, ls = 1, dls = 0)
#     xlearner_loss, xexpert_loss, xlearner_preds,weight_L  = AA_Class_discounted(outcomes, prediction_array,cmb,1,1)
#     x3 = xlearner_preds * data_datesum['NetUsdPnL'].values
#     AA_list.append(x3)
#     print('Trial {} complete'.format(cmb))

#     cmb = CombinedLoss(return_scale = 100, ls = 2, dls = 1)
#     xlearner_loss, xexpert_loss, xlearner_preds,weight_L  = AA_Class_discounted(outcomes, prediction_array,cmb,1,1)
#     x4 = xlearner_preds * data_datesum['NetUsdPnL'].values
#     AA_list.append(x4)
#     print('Trial {} complete'.format(cmb))
    
#     cmb = CombinedLoss(return_scale = 1, ls = 0, dls = 1)
#     xlearner_loss, xexpert_loss, xlearner_preds,weight_L  = AA_Class_discounted(outcomes, prediction_array,cmb,1,1)
#     x5 = xlearner_preds * data_datesum['NetUsdPnL'].values
#     AA_list.append(x5)
#     print('Trial {} complete'.format(cmb))
    
#     cmb = CombinedLoss(return_scale = 50, ls = 0, dls = 1)
#     xlearner_loss, xexpert_loss, xlearner_preds,weight_L  = AA_Class_discounted(outcomes, prediction_array,cmb,1,1)
#     x6 = xlearner_preds * data_datesum['NetUsdPnL'].values
#     AA_list.append(x6)
#     print('Trial {} complete'.format(cmb))
    
#     cmb = CombinedLoss(return_scale = 100, ls = 1, dls = 1)
#     xlearner_loss, xexpert_loss, xlearner_preds,weight_L  = AA_Class_discounted(outcomes, prediction_array,cmb,1,1)
#     x7 = xlearner_preds * data_datesum['NetUsdPnL'].values
#     AA_list.append(x7)
#     print('Trial {} complete'.format(cmb))
    
#     cmb = CombinedLoss(return_scale = 1, ls = 1, dls = 0)
#     xlearner_loss, xexpert_loss, xlearner_preds,weight_L  = AA_Class_discounted(outcomes, prediction_array,cmb,1,0.1)
#     x8 = xlearner_preds * data_datesum['NetUsdPnL'].values
#     AA_list.append(x8)
#     print('Trial {} complete'.format(cmb))
    
#     # cmb = CombinedLoss(return_scale = 1, ls = 1, dls = 0)
#     # xlearner_loss, xexpert_loss, xlearner_preds,weight_L  = AA_Class_discounted(outcomes, prediction_array,cmb,1,0.1)
#     # x9 = xlearner_preds * data_datesum['NetUsdPnL'].values
#     # AA_list.append(x9)
#     # print('Trial {} complete'.format(cmb))
    
#     # cmb = CombinedLoss(return_scale = 1, ls = 1, dls = 0)
#     # xlearner_loss, xexpert_loss, xlearner_preds,weight_L  = AA_Class_discounted(outcomes, prediction_array,cmb,1,0.01)
#     # x11 = xlearner_preds * data_datesum['NetUsdPnL'].values
#     # AA_list.append(x11)
#     # print('Trial {} complete'.format(cmb))
    
#     # cmb = CombinedLoss(return_scale = 1, ls = 1, dls = 0)
#     # xlearner_loss, xexpert_loss, xlearner_preds,weight_L  = AA_Class_discounted(outcomes, prediction_array,cmb,1,0.001)
#     # x12 = xlearner_preds * data_datesum['NetUsdPnL'].values
#     # AA_list.append(x12)
#     # print('Trial {} complete'.format(cmb))

#     # cmb = CombinedLoss(return_scale = 1, ls = 1, dls = 0)
#     # xlearner_loss, xexpert_loss, xlearner_preds,weight_L  = AA_Class_discounted(outcomes, prediction_array,cmb,1,1)
#     # x13 = xlearner_preds * data_datesum['NetUsdPnL'].values
#     # AA_list.append(x13)
#     # print('Trial {} complete'.format(cmb))
    
# #     outcomes = data_datesum['NetUsdPnL'].values
# # # # #     #outcomes = outcomes[:1000]
#     outcomes = data_datesum['NetUsdPnL'].values
    
#     cmb = PnL_weak_loss(return_scale = 1)
#     xlearner_loss, xexpert_loss, xlearner_preds, weights  = weak_AA_class(outcomes, prediction_array,cmb,2000000)
#     x10 = xlearner_preds * data_datesum['NetUsdPnL'].values
#     AA_list.append(x10)
#     print('Trial {} complete'.format(cmb))
    
#     cmb = LS_pnl_loss(return_scale = 0.000001, ls = 1, dls = 0)
#     xlearner_loss, xexpert_loss, xlearner_preds,weight_L  = AA_Class_discounted(outcomes, prediction_array,cmb,1,1)
#     x14 = xlearner_preds * data_datesum['NetUsdPnL'].values
#     AA_list.append(x14)
#     print('Trial {} complete'.format(cmb))
    
#     cmb = LS_pnl_loss(return_scale = 0.000001, ls = 1, dls = 1)
#     xlearner_loss, xexpert_loss, xlearner_preds,weight_L  = AA_Class_discounted(outcomes, prediction_array,cmb,1,1)
#     x15 = xlearner_preds * data_datesum['NetUsdPnL'].values
#     AA_list.append(x15)
#     print('Trial {} complete'.format(cmb))
    

    
#    # learner_HF = learner_hedge_fraction(client_root,client_wealth, x13, weight_L)#     
# #     #cmb = PnLLoss(return_scale = 1, cof = 0.0001)
# #     #learner_loss, expert_loss, learner_preds = AA_Class(outcomes, prediction_array,cmb,1)
# #     
# #     #cmb = PnL_weak_loss(return_scale = 1)
# #     
# #     #xlearner_loss, xexpert_loss, xlearner_preds, weights  = weak_AA_class(outcomes, prediction_array,cmb,2000000)
# #     #xlearner_loss, xexpert_loss, xlearner_preds,weight_L  = AA_Class_discounted(outcomes, prediction_array,cmb,1,0.05)
# #    # wlearner_loss, wexpert_loss, wlearner_preds = AA(outcomes_binary, prediction_array, absoloute_loss,2)
# # 
# #     #plt.figure()
# #     #(data_datesum['NetUsdPnL'].cumsum() - data_datesum['NetUsdPnL'].cumsum().cummax() ).plot()
# #     
# #     
# #     
# #     
#     pnl = []
    
#     pnl.append(sum(data_datesum['NetUsdPnL'].values))
#     pnl.append(sum(data_datesum['NetUsdPnL'].values + x0))
#     pnl.append(sum(data_datesum['NetUsdPnL'].values + x1))
#     pnl.append(sum(data_datesum['NetUsdPnL'].values + x2))
#     pnl.append(sum(data_datesum['NetUsdPnL'].values + x3))
#     pnl.append(sum(data_datesum['NetUsdPnL'].values + x4))
#     pnl.append(sum(data_datesum['NetUsdPnL'].values + x5))
#     pnl.append(sum(data_datesum['NetUsdPnL'].values + x6))
#     pnl.append(sum(data_datesum['NetUsdPnL'].values + x7))
#     pnl.append(sum(data_datesum['NetUsdPnL'].values + x8))
#     pnl.append(sum(data_datesum['NetUsdPnL'].values + x14))
#     pnl.append(sum(data_datesum['NetUsdPnL'].values + x15))
#     pnl.append(sum(data_datesum['NetUsdPnL'].values + x10))

    
#     DD = []
#     calmar = []

    
#     dd = drawdown(data_datesum['NetUsdPnL'])
#     DD.append(dd.min())
#     calmar.append(pnl[0] / abs(DD[0]))
    
#     dataf = pd.DataFrame((data_datesum['NetUsdPnL'].values + x0))
#     dd = drawdown(dataf)
#     DD.append(dd.min())
#     calmar.append(pnl[1] / abs(DD[1]))
    
#     dataf = pd.DataFrame((data_datesum['NetUsdPnL'].values + x1))
#     dd = drawdown(dataf)
#     DD.append(dd.min())
#     calmar.append(pnl[2] / abs(DD[2]))


#     dataf = pd.DataFrame((data_datesum['NetUsdPnL'].values + x2))
#     dd = drawdown(dataf)
#     DD.append(dd.min())
#     calmar.append(pnl[3] / abs(DD[3]))
    
#     dataf = pd.DataFrame((data_datesum['NetUsdPnL'].values + x3))
#     dd = drawdown(dataf)
#     DD.append(dd.min())
#     calmar.append(pnl[4] / abs(DD[4]))
    
#     dataf = pd.DataFrame((data_datesum['NetUsdPnL'].values + x4))
#     dd = drawdown(dataf)
#     DD.append(dd.min())
#     calmar.append(pnl[5] / abs(DD[5]))
    
    
#     dataf = pd.DataFrame((data_datesum['NetUsdPnL'].values + x5))
#     dd = drawdown(dataf)
#     DD.append(dd.min())
#     calmar.append(pnl[6] / abs(DD[6]))
    
#     dataf = pd.DataFrame((data_datesum['NetUsdPnL'].values + x6))
#     dd = drawdown(dataf)
#     DD.append(dd.min())
#     calmar.append(pnl[7] / abs(DD[7]))
    
    
#     dataf = pd.DataFrame((data_datesum['NetUsdPnL'].values + x7))
#     dd = drawdown(dataf)
#     DD.append(dd.min())
#     calmar.append(pnl[8] / abs(DD[8]))
    
    
#     dataf = pd.DataFrame((data_datesum['NetUsdPnL'].values + x8))
#     dd = drawdown(dataf)
#     DD.append(dd.min())
#     calmar.append(pnl[9] / abs(DD[9]))
    
    
#     dataf = pd.DataFrame((data_datesum['NetUsdPnL'].values + x10))
#     dd = drawdown(dataf)
#     DD.append(dd.min())
#     calmar.append(pnl[10] / abs(DD[10]))

#     dataf = pd.DataFrame((data_datesum['NetUsdPnL'].values + x14))
#     dd = drawdown(dataf)
#     DD.append(dd.min())
#     calmar.append(pnl[11] / abs(DD[11]))
    
#     dataf = pd.DataFrame((data_datesum['NetUsdPnL'].values + x15))
#     dd = drawdown(dataf)
#     DD.append(dd.min())
#     calmar.append(pnl[12] / abs(DD[12]))
    
#     plt.figure()
#     plt.plot(np.cumsum(data_datesum['NetUsdPnL'].values))
#     #plt.plot(np.cumsum(x2))
#     plt.plot(np.cumsum(data_datesum['NetUsdPnL'].values + x0))
#     plt.plot(np.cumsum(data_datesum['NetUsdPnL'].values + x1))
#     plt.plot(np.cumsum(data_datesum['NetUsdPnL'].values + x2))
#     plt.plot(np.cumsum(data_datesum['NetUsdPnL'].values + x3))
#     plt.plot(np.cumsum(data_datesum['NetUsdPnL'].values + x4))
#     plt.plot(np.cumsum(data_datesum['NetUsdPnL'].values + x5))
#     plt.plot(np.cumsum(data_datesum['NetUsdPnL'].values + x6))
#     plt.plot(np.cumsum(data_datesum['NetUsdPnL'].values + x7))
#     plt.plot(np.cumsum(data_datesum['NetUsdPnL'].values + x8))
#     #plt.plot(np.cumsum(data_datesum['NetUsdPnL'].values + x9))
#     plt.plot(np.cumsum(data_datesum['NetUsdPnL'].values + x10))
#     plt.plot(np.cumsum(data_datesum['NetUsdPnL'].values + x14))
#     plt.plot(np.cumsum(data_datesum['NetUsdPnL'].values + x15))
#     # plt.plot(np.cumsum(data_datesum['NetUsdPnL'].values + x11))
#     # plt.plot(np.cumsum(data_datesum['NetUsdPnL'].values + x12))
#     # plt.plot(np.cumsum(data_datesum['NetUsdPnL'].values + x13))
#     plt.xlabel('Time', fontsize = 40)
#     plt.ylabel('PnL', fontsize = 40)
    
    
#     plt.legend(['Client', 'Equal','R=1, LS=1, DLS=0', 'R=50, LS=1, DLS=0',
#                 'R=100, LS=1, DLS=0', 'R=100, LS=2, DLS=1', 'R=1, LS=0, DLS=1',
#                 'R=50, LS=0, DLS=1', 'R=100, LS=1, DLS=1',
#                 'R=1, LS=1, DLS=0, DF=0.1', 'WAA', 'PnL, LS=1, DLS=0',
#                 'PnL, LS=1, DLS=1'])
    
    
    

#     AA_Name = ['Equal','R=1, LS=1, DLS=0', 'R=50, LS=1, DLS=0',
#                 'R=100, LS=1, DLS=0', 'R=100, LS=2, DLS=1', 'R=1, LS=0, DLS=1',
#                 'R=50, LS=0, DLS=1', 'R=100, LS=1, DLS=1',
#                 'R=1, LS=1, DLS=0, DF=0.1', 'WAA', 'PnL, LS=1, DLS=0',
#                 'PnL, LS=1, DLS=1']
    
#     # # plt.legend(['Client', '1','0.9', '0.8','0.7','0.6','0.5','0.3','0.2','0.1', 'WAA', '0.01','0.001','0.0001'])
    
    
    

#     # # AA_Name = ['1','0.9', '0.8','0.7','0.6','0.5','0.3','0.2','0.1', 'WAA','0.01','0.001','0.0001']
    
    
      
#     plot_DD(prediction_array, data_datesum['NetUsdPnL'].values, AA_list, AA_Name)
 
#     names = ['Client', 'Equal','R=1, LS=1, DLS=0', 'R=50, LS=1, DLS=0',
#                   'R=100, LS=1, DLS=0', 'R=100, LS=2, DLS=1', 'R=1, LS=0, DLS=1',
#                   'R=50, LS=0, DLS=1', 'R=100, LS=1, DLS=1',
#                   'R=1, LS=1, DLS=0, DF=0.1', 'PnL, LS=1, DLS=0',
#                 'PnL, LS=1, DLS=1','WAA']
   
#     info = {'AA Parameters': names,
#             'PnL': pnl,
#             'Drawdown': DD,
#             'Calmar': calmar}
#     print(tabulate(info, headers='keys', tablefmt='fancy_grid'))
 
 







   
#     
#    # y = x + data_datesum['NetUsdPnL'].values
#     #plt.plot(np.cumsum(x2 + data_datesum['NetUsdPnL'].values))
#     #x = Xlearner_preds * data_datesum['NetUsdPnL'].values    #x[1] = 0
#     #plt.plot(x.cumsum())
#    # yweak = x + data_datesum['NetUsdPnL'].values
#     #lt.figure()
#     #data_datesum['NetUsdPnL'].cumsum() - data_datesum['NetUsdPnL'].cumsum().cummax() ).plot()
#     #(pd.DataFrame(y, index = data_datesum['NetUsdPnL'].index).cumsum() - pd.DataFrame(y, index = data_datesum['NetUsdPnL'].index).cumsum().cummax() ).plot()
#     #(pd.DataFrame(yweak, index = data_datesum['NetUsdPnL'].index).cumsum() - pd.DataFrame(yweak, index = data_datesum['NetUsdPnL'].index).cumsum().cummax() ).plot()
#     
#     
#     #ax = data_datesum['NetUsdPnL'].cumsum().plot(fontsize = 20)
#     #test2 = pd.DataFrame(x2, index = data_datesum['NetUsdPnL'].index)
#     #test2.cumsum().plot(ax=ax)
#     #test2['net'] = test2[0].values + data_datesum['NetUsdPnL'].values
#     #test2['net'].cumsum().plot(ax=ax)
# 
# =============================================================================
    
    