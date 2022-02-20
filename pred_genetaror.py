# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 00:00:04 2021

@author: Owner
"""
import pandas as pd

from hedge_project import *


if __name__ == '__main__':
    data = pd.read_csv(r"C:\Users\Owner\Documents\University\PhD\Data\2021\Fudged_1hour_2014To2016.csv")
    data['QdfTime'] = pd.to_datetime(data['QdfTime'])
    data_eur_usd = data[data['Symbol'] == 'USD/CHF']
    data_eur_usd['mid_price'] = data_eur_usd['Mid']
    data_datesum = data_eur_usd.groupby(['QdfTime']).sum()
    #data_datesum = data_datesum.loc['2015-02-01':'2016-04-27']
    
    #ranges:
        #limits = [0,100]
        #hedge_fractions = [-1,0]
        #skew = [0,1]
        #window = [-24*60, 24*60]
    limits = (0.1, 0.2, 0.8, 1, 2, 4, 5, 6, 10, 20,40)
    hedge_fractions = (0.1, 0.15, 0.25, 0.3, 0.4, 0.5, 0.75, 0.85, 0.9, 1)
    skew = (0.1, 0.2,0.3, 0.5,0.6,0.8, 0.9)
    window = (-24*60,-24*30,-24*14,-24*7,-24,-12,-6,-3,3, 6, 12, 24, 24*7, 24*14,24*30,24*60)
    
    indicators = {}
    for w in window:
        indicators[w] = indicator(data_datesum['mid_price'].values,w)
    
    
    
    hedge_array =  hedge_experts_skew(limits, hedge_fractions, skew,window)
    
    agents = []
    for s in hedge_array:
        agents.append(Agent(limits=s[0],hedge_fractions=s[1],skew = s[2],indicator=indicators[s[3]],window= s[3]))
    
      
    prediction_array = grid_agents(agents,data_datesum['NetPosUsd'].values)
    
    pd.to_pickle(data_datesum, r"C:\Users\Owner\Documents\University\PhD\Data\2021_data\preds_for_AA\data_chf_usd.obj")
    pd.to_pickle(agents, r"C:\Users\Owner\Documents\University\PhD\Data\2021_data\preds_for_AA\agents_chf_usd.obj")
    pd.to_pickle(prediction_array, r"C:\Users\Owner\Documents\University\PhD\Data\2021_data\preds_for_AA\predictions_chf_usd.obj")