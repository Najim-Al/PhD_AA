# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 13:56:01 2021

@author: Owner
"""
import numpy as np
import math
import  pandas as pd
from LossFunction import *

def squared_loss(outcome, prediction):
    loss = (prediction - outcome)**2
    return loss

def absoloute_loss(outcome, prediction):
    loss = abs(prediction - outcome)
    return loss

# AA's have equal inital experts weights 

def AA(outcomes, prediction, loss, learning_rate):
    T = len(outcomes)
    N_experts = np.size(prediction,1)
    weights = np.ones(N_experts)
    learner_prediction = np.zeros(T)
    loss_log = np.zeros((T, N_experts))
    learner_loss = np.zeros(T)
    for t in range(T):
        print(t)
        norm_weights = weights / sum(weights)
        learner_prediction[t] = sum( norm_weights * prediction[t] )
        learner_loss[t] = loss(outcomes[t],learner_prediction[t])
        for i in range(N_experts):
            loss_log[t,i] = loss(outcomes[t],prediction[t,i])
            weights[i] = weights[i] * np.exp(-learning_rate*loss_log[t,i]) #+ 0.5    
    return learner_loss, loss_log, learner_prediction

def AA_Class(outcomes, prediction, loss: LossFunc, learning_rate):
    T = len(outcomes)
    N_experts = np.size(prediction,1)
    weights = np.ones(N_experts)
    learner_prediction = np.zeros(T)
    loss_log = np.zeros((T, N_experts))
    weight_log = np.zeros((T, N_experts))
    learner_loss = np.zeros(T)
    for t in range(T):
      #  if np.isnan(sum(weights)):
       #     break
       # print(t)
#        if t % 1000 == 0:
#            weights = np.ones(N_experts)
        norm_weights = weights / sum(weights)
        learner_prediction[t] = sum( norm_weights * prediction[t] )
        learner_loss[t] = loss.calc_loss(outcomes[t],learner_prediction[t]) 
        for i in range(N_experts):
            loss_log[t,i] = loss.calc_loss(outcomes[t],prediction[t,i]) 
            weights[i] = weights[i]  * np.exp(-learning_rate*loss_log[t,i]) #+ 0.5    
            if np.isnan(weights[i]):
                weights[i] = 0
            elif weights[i] <= 0:
                weights[i] = 0
        weight_log[t] = weights
            
    return learner_loss, loss_log, learner_prediction ,weight_log

def AA_Class_wealth(outcomes, prediction, loss: LossFunc, learning_rate):
    T = len(outcomes)
    N_experts = np.size(prediction,1)
    weights = np.ones(N_experts)
    learner_prediction = np.zeros(T)
    loss_log = np.zeros((T, N_experts))
    weight_log = np.zeros((T, N_experts))
    learner_loss = np.zeros(T)
    expert_wealth = np.ones(N_experts)
    expert_wealth = expert_wealth*10000000
    learner_wealth = 10000000
    wealth_log = np.zeros((T, N_experts))
    for t in range(T):
       
      #  if np.isnan(sum(weights)):
       #     break
       # print(t)
#        if t % 1000 == 0:
#            weights = np.ones(N_experts)
        norm_weights = weights / sum(weights)
        
        learner_prediction[t] = sum( norm_weights * prediction[t] )
        learner_loss[t] = loss.calc_loss(outcomes[t],learner_prediction[t],learner_wealth ) 
        learner_wealth += outcomes[t] * (1 + learner_prediction[t])
        for i in range(N_experts):
            
            loss_log[t,i] = loss.calc_loss(outcomes[t],prediction[t,i], expert_wealth[i]) 
            weights[i] = weights[i]  * np.exp(-learning_rate*loss_log[t,i]) #+ 0.5    
            expert_wealth[i] += outcomes[t] * (1 + prediction[t,i])
            wealth_log[t,i] =  expert_wealth[i]
            if np.isnan(weights[i]):
                print('nan')
                weights[i] = 0
            elif weights[i] <= 0:
                print('zero')
                weights[i] = 0
        weight_log[t] = weights
        
    return learner_loss, loss_log, learner_prediction ,weight_log,wealth_log
    
def AA_Class_wealth_FTL(outcomes, prediction, loss: LossFunc, learning_rate):
    T = len(outcomes)
    N_experts = np.size(prediction,1)
    weights = np.ones(N_experts)
    learner_prediction = np.zeros(T)
    loss_log = np.zeros((T, N_experts))
    weight_log = np.zeros((T, N_experts))
    learner_loss = np.zeros(T)
    expert_wealth = np.ones(N_experts)
    expert_wealth = expert_wealth*10000000
    learner_wealth = 10000000
    wealth_log = np.zeros((T, N_experts))
    idx = 0
    for t in range(T):
        
      #  if np.isnan(sum(weights)):
       #     break
       # print(t)
#        if t % 1000 == 0:
#            weights = np.ones(N_experts)
        learner_prediction[t] =  prediction[t,idx] 
        learner_loss[t] = loss.calc_loss(outcomes[t],learner_prediction[t],learner_wealth ) 
        learner_wealth += outcomes[t] * (1 + learner_prediction[t])
        for i in range(N_experts):
            loss_log[t,i] = loss.calc_loss(outcomes[t],prediction[t,i], expert_wealth[i]) 
            weights[i] = weights[i]  * np.exp(-learning_rate*loss_log[t,i]) #+ 0.5    
            expert_wealth[i] += outcomes[t] * (1 + prediction[t,i])
            wealth_log[t,i] =  expert_wealth[i]
            if np.isnan(weights[i]):
                weights[i] = 0
            elif weights[i] <= 0:
                weights[i] = 0
        idx = weights.argmax()
        print(idx)
        weight_log[t] = weights
        
            
    return learner_loss, loss_log, learner_prediction ,weight_log,wealth_log 


def AA_equal(outcomes, prediction, loss: LossFunc, learning_rate):
    T = len(outcomes)
    N_experts = np.size(prediction,1)
    weights = np.ones(N_experts)
    learner_prediction = np.zeros(T)
    loss_log = np.zeros((T, N_experts))
    weight_log = np.zeros((T, N_experts))
    learner_loss = np.zeros(T)
    #learner_wealth = 10000000
    for t in range(T):
      #  if np.isnan(sum(weights)):
       #     break
       # print(t)
#        if t % 1000 == 0:
#            weights = np.ones(N_experts)
        norm_weights = weights / sum(weights)
        learner_prediction[t] = sum( norm_weights * prediction[t] )
        #learner_wealth += outcomes[t] * (1 + learner_prediction[t])
        learner_loss[t] = loss.calc_loss(outcomes[t],learner_prediction[t]) 
        #learner_wealth += outcomes[t] * (1 + learner_prediction[t])

        weight_log[t] = weights
            
    return learner_loss, loss_log, learner_prediction ,weight_log

def AA_Class_multi(outcomes, prediction, loss: LossFunc, learning_rate):
    T = len(outcomes[:,0])
    S = len(outcomes[0,:])
    N_experts = np.size(prediction,1)
    weights = np.ones(N_experts)
    learner_prediction = np.zeros((T,S))
    loss_log = np.zeros((T, N_experts))
    weight_log = np.zeros((T, N_experts))
    learner_loss = np.zeros(T)
    for t in range(T):
      #  if np.isnan(sum(weights)):
       #     break
        #print(t)
#        if t % 1000 == 0:
#            weights = np.ones(N_experts)
        norm_weights = weights / sum(weights)
        learner_prediction[t,:] = np.sum( norm_weights * np.array(prediction[t]).T,axis=1)
        learner_loss[t] = loss.calc_loss(outcomes[t],learner_prediction[t]) 
        for i in range(N_experts):
            loss_log[t,i] = loss.calc_loss(outcomes[t],prediction[t][i]) 
            weights[i] = weights[i]  * np.exp(-learning_rate*loss_log[t,i]) #+ 0.5    
            if weights[i] <= 0:
                weights[i] = 0
            elif np.isnan(weights[i]):
                weights[i] = 0
        weight_log[t] = weights
            
    return learner_loss, loss_log, learner_prediction ,weight_log

def weak_AA_class_multi(outcomes, prediction, loss: LossFunc, Upper_Bound):
    T = len(outcomes)
    S = len(outcomes[0,:])
    N_experts = np.size(prediction,1)
    norm_weights = np.ones(N_experts)
    learner_prediction = np.zeros((T,S))
    loss_log = np.zeros((T, N_experts))
    cumsum_loss = np.zeros(N_experts)
    learner_loss = np.zeros(T)
    weight_log = np.zeros((T, N_experts))
    C = (2 * math.sqrt(N_experts)) / Upper_Bound
    for t in range(T):
        #print(t)
        #if t % 10000 == 0:
         #   cumsum_loss = np.zeros(N_experts)
        learner_prediction[t,:] = np.sum( norm_weights * np.array(prediction[t]).T,axis=1)
        learner_loss[t] = loss.calc_loss(outcomes[t],learner_prediction[t])
        denominator = 0
        learning_rate = C / math.sqrt(t)
        for j in range(N_experts):
                denominator +=  np.exp(-learning_rate*cumsum_loss[j])
        for i in range(N_experts):
            norm_weights[i] = (np.exp(-learning_rate*cumsum_loss[i])) / denominator
            weight_log[t,i] = norm_weights[i]
            loss_log[t,i] = loss.calc_loss(outcomes[t],prediction[t][i])
        cumsum_loss += loss_log[t,:]
    return learner_loss, loss_log, learner_prediction, weight_log



def AA_Class_discounted(outcomes, prediction, loss: LossFunc, learning_rate, discounting_factor):
    T = len(outcomes)
    N_experts = np.size(prediction,1)
    weights = np.ones(N_experts)
    learner_prediction = np.zeros(T)
    loss_log = np.zeros((T, N_experts))
    weight_log = np.zeros((T, N_experts))
    learner_loss = np.zeros(T)
    for t in range(T):
        if np.isnan(sum(weights)):
            break
        #print(t)
#        if t % 1000 == 0:
#            weights = np.ones(N_experts)
        norm_weights = weights / sum(weights)
        weight_log[t] = norm_weights
        learner_prediction[t] = sum( norm_weights * prediction[t] )
        learner_loss[t] = loss.calc_loss(outcomes[t],learner_prediction[t]) 
        for i in range(N_experts):
            loss_log[t,i] = loss.calc_loss(outcomes[t],prediction[t,i]) 
            weights[i] = (weights[i]**discounting_factor) * np.exp(-learning_rate*loss_log[t,i]) #+ 0.5    
            if np.isnan(weights[i]):
                weights[i] = 0
            elif weights[i] <= 0:
                weights[i] = 0
            
    return learner_loss, loss_log, learner_prediction,weight_log

def weak_AA_class(outcomes, prediction, loss: LossFunc, Upper_Bound):
    T = len(outcomes)
    N_experts = np.size(prediction,1)
    norm_weights = np.ones(N_experts)
    learner_prediction = np.zeros(T)
    loss_log = np.zeros((T, N_experts))
    cumsum_loss = np.zeros(N_experts)
    learner_loss = np.zeros(T)
    weight_log = np.zeros((T, N_experts))
    C = (2*math.sqrt(N_experts)) / Upper_Bound
    for t in range(T):
        #print(t)
        #if t % 10000 == 0:
         #   cumsum_loss = np.zeros(N_experts)
        learner_prediction[t] = sum( norm_weights * prediction[t] )
        learner_loss[t] = loss.calc_loss(outcomes[t],learner_prediction[t])
        denominator = 0
        learning_rate = C / math.sqrt(max(t,1))
        for j in range(N_experts):
                denominator +=  np.exp(-learning_rate*cumsum_loss[j])
        for i in range(N_experts):
            norm_weights[i] = (np.exp(-learning_rate*cumsum_loss[i])) / denominator
            weight_log[t,i] = np.exp(-learning_rate*cumsum_loss[i])
            loss_log[t,i] = loss.calc_loss(outcomes[t],prediction[t,i])
            cumsum_loss[i] += loss_log[t,i]
    return learner_loss, loss_log, learner_prediction, weight_log


# def weak_AA_class_discounting(outcomes, prediction, loss: LossFunc, Upper_Bound,alpha):
#     T = len(outcomes)
#     N_experts = np.size(prediction,1)
#     norm_weights = np.ones(N_experts)
#     learner_prediction = np.zeros(T)
#     loss_log = np.zeros((T, N_experts))
#     cumsum_loss = np.zeros(N_experts)
#     learner_loss = np.zeros(T)
#     weight_log = np.zeros((T, N_experts))
#     norm_weight_log = np.zeros((T, N_experts))
#     C = (2*math.sqrt(N_experts)) / Upper_Bound
#     for t in range(T):
#         #print(t)
#         #if t % 10000 == 0:
#          #   cumsum_loss = np.zeros(N_experts)
#         learner_prediction[t] = sum( norm_weights * prediction[t] )
#         learner_loss[t] = loss.calc_loss(outcomes[t],learner_prediction[t])
#         denominator = 0
#         learning_rate = C / math.sqrt(max(t,1))
#         for j in range(N_experts):
#             denominator +=  np.exp(-learning_rate*alpha*cumsum_loss[j])
#             loss_log[t,j] = loss.calc_loss(outcomes[t],prediction[t,j])
#         for i in range(N_experts):
#             norm_weights[i] = (np.exp(-learning_rate*alpha*cumsum_loss[i])) / denominator
#             weight_log[t,i] = np.exp(-learning_rate*alpha*cumsum_loss[i])
#             norm_weight_log[t,i] = norm_weights[i]
#             #loss_log[t,i] = loss.calc_loss(outcomes[t],prediction[t,i])
#             cumsum_loss[i] =  (alpha*cumsum_loss[i]) + (loss_log[t,i]) 
#     return learner_loss, loss_log, learner_prediction, weight_log, norm_weight_log


def weak_AA_class_discounting(outcomes, prediction, loss: LossFunc, Upper_Bound,alpha):
    T = len(outcomes)
    N_experts = np.size(prediction,1)
    norm_weights = np.ones(N_experts)
    learner_prediction = np.zeros(T)
    loss_log = np.zeros((T, N_experts))
    cumsum_loss = np.zeros(N_experts)
    learner_loss = np.zeros(T)
    weight_log = np.zeros((T, N_experts))
    norm_weight_log = np.zeros((T, N_experts))
    C = (2*math.sqrt(N_experts)) / Upper_Bound
    for t in range(T):
        #print(t)
        #if t % 10000 == 0:
          #   cumsum_loss = np.zeros(N_experts)
        learner_prediction[t] = sum( norm_weights * prediction[t] )
        learner_loss[t] = loss.calc_loss(outcomes[t],learner_prediction[t])
        denominator = 0
        learning_rate = C / math.sqrt(max(t,1))
        # loss_min = 0
        for j in range(N_experts):
            denominator +=  np.exp(-learning_rate*alpha*cumsum_loss[j])
            loss_log[t,j] = loss.calc_loss(outcomes[t],prediction[t,j])
            # if loss_log[t,j] < loss_min:
            #     loss_min = loss_log[t,j]
        for i in range(N_experts):
            norm_weights[i] = (np.exp(-learning_rate*alpha*cumsum_loss[i])) / denominator
            weight_log[t,i] = np.exp(-learning_rate*alpha*cumsum_loss[i])
            norm_weight_log[t,i] = norm_weights[i]
            #loss_log[t,i] = loss.calc_loss(outcomes[t],prediction[t,i])
            cumsum_loss[i] =  (alpha*cumsum_loss[i]) + (loss_log[t,i]) #- loss_min
    return learner_loss, loss_log, learner_prediction, weight_log, norm_weight_log

def weak_AA_class_discounting_min(outcomes, prediction, loss: LossFunc, Upper_Bound,alpha):
    T = len(outcomes)
    N_experts = np.size(prediction,1)
    norm_weights = np.ones(N_experts)
    learner_prediction = np.zeros(T)
    loss_log = np.zeros((T, N_experts))
    cumsum_loss = np.zeros(N_experts)
    learner_loss = np.zeros(T)
    weight_log = np.zeros((T, N_experts))
    norm_weight_log = np.zeros((T, N_experts))
    C = (2*math.sqrt(N_experts)) / Upper_Bound
    for t in range(T):
        #print(t)
        #if t % 10000 == 0:
          #   cumsum_loss = np.zeros(N_experts)
        learner_prediction[t] = sum( norm_weights * prediction[t] )
        learner_loss[t] = loss.calc_loss(outcomes[t],learner_prediction[t])
        denominator = 0
        learning_rate = C / math.sqrt(max(t,1))
        loss_min = 0
        for j in range(N_experts):
            denominator +=  np.exp(-learning_rate*alpha*cumsum_loss[j])
            loss_log[t,j] = loss.calc_loss(outcomes[t],prediction[t,j])
            if loss_log[t,j] < loss_min:
                loss_min = loss_log[t,j]
        for i in range(N_experts):
            norm_weights[i] = (np.exp(-learning_rate*alpha*cumsum_loss[i])) / denominator
            weight_log[t,i] = np.exp(-learning_rate*alpha*cumsum_loss[i])
            norm_weight_log[t,i] = norm_weights[i]
            #loss_log[t,i] = loss.calc_loss(outcomes[t],prediction[t,i])
            cumsum_loss[i] =  (alpha*cumsum_loss[i]) + (loss_log[t,i]) - loss_min
    return learner_loss, loss_log, learner_prediction, weight_log, norm_weight_log


# def weak_AA_class_discounting(outcomes, prediction, loss: LossFunc, Upper_Bound,alpha):
#     T = len(outcomes)
#     N_experts = np.size(prediction,1)
#     norm_weights = np.ones(N_experts)
#     learner_prediction = np.zeros(T)
#     loss_log = np.zeros((T, N_experts))
#     cumsum_loss = np.zeros(N_experts)
#     learner_loss = np.zeros(T)
#     weight_log = np.zeros((T, N_experts))
#     C = (2*math.sqrt(N_experts)) / Upper_Bound
#     for t in range(T):
#         #print(t)
#         #if t % 10000 == 0:
#           #   cumsum_loss = np.zeros(N_experts)
#         learner_prediction[t] = sum( norm_weights * prediction[t] )
#         learner_loss[t] = loss.calc_loss(outcomes[t],learner_prediction[t])
#         denominator = 0
#         learning_rate = C / math.sqrt(max(t,1))
#         for j in range(N_experts):
#                 denominator +=  np.exp(-learning_rate*cumsum_loss[j])
#         for i in range(N_experts):
#             norm_weights[i] = (np.exp(-learning_rate*cumsum_loss[i])) / denominator
#             weight_log[t,i] = np.exp(-learning_rate*cumsum_loss[i])
#             loss_log[t,i] = loss.calc_loss(outcomes[t],prediction[t,i])
#             cumsum_loss[i] =  (alpha*cumsum_loss[i]) +loss_log[t,i]
#     return learner_loss, loss_log, learner_prediction, weight_log

def weak_AA_class_raw(outcomes, prediction, loss: LossFunc, Upper_Bound):
    T = len(outcomes)
    N_experts = np.size(prediction,1)
    norm_weights = np.ones(N_experts)
    learner_prediction = np.zeros(T)
    loss_log = np.zeros((T, N_experts))
    cumsum_loss = np.zeros(N_experts)
    cumsum_loss_norm = np.zeros(N_experts)
    learner_loss = np.zeros(T)
    weight_log = np.zeros((T, N_experts))
    C = 1 #(2*math.sqrt(N_experts)) / Upper_Bound
    for t in range(T):
        #print(t)
        #if t % 10000 == 0:
         #   cumsum_loss = np.zeros(N_experts)
        learner_prediction[t] = sum( norm_weights * prediction[t] )
        learner_loss[t] = loss.calc_loss(outcomes[t],learner_prediction[t])
        denominator = 0
        learning_rate = C / math.sqrt(max(t,1))
        
        for j in range(N_experts):
                denominator +=  np.exp(-learning_rate*cumsum_loss_norm[j])
        for i in range(N_experts):
            norm_weights[i] = (np.exp(-learning_rate*cumsum_loss_norm[i])) / denominator
            weight_log[t,i] = (np.exp(-learning_rate*cumsum_loss_norm[i])) #norm_weights[i]
            loss_log[t,i] = loss.calc_loss(outcomes[t],prediction[t,i])
            cumsum_loss[i] += loss_log[t,i]
        cumsum_loss_norm = cumsum_loss/sum(cumsum_loss)
    return learner_loss, loss_log, learner_prediction, weight_log


def weak_AA_class_wealth(outcomes, prediction, loss: LossFunc, Upper_Bound):
    T = len(outcomes)
    N_experts = np.size(prediction,1)
    norm_weights = np.ones(N_experts)
    norm_weights = norm_weights/ N_experts
    learner_prediction = np.zeros(T)
    loss_log = np.zeros((T, N_experts))
    cumsum_loss = np.zeros(N_experts)
    expert_wealth = np.ones(N_experts)
    expert_wealth = expert_wealth*1000000
    learner_wealth = 1000000
    learner_loss = np.zeros(T)
    weight_log = np.zeros((T, N_experts))
    wealth_log = np.zeros((T, N_experts))
    C = (math.sqrt(np.log10(N_experts))) / Upper_Bound
    for t in range(T):
        #print(t)
        #if t % 10000 == 0:
         #   cumsum_loss = np.zeros(N_experts)
        learner_prediction[t] = sum( norm_weights * prediction[t] )
        learner_loss[t] = loss.calc_loss(outcomes[t],learner_prediction[t], learner_wealth)
        learner_wealth += outcomes[t] * (1 + learner_prediction[t])
        denominator = 0
        learning_rate = C / math.sqrt(max(t,1))
        for j in range(N_experts):
                denominator +=  np.exp(learning_rate)**cumsum_loss[j]
        for i in range(N_experts):
            norm_weights[i] = (np.exp(learning_rate)**cumsum_loss[i]) / denominator
            weight_log[t,i] = (np.exp(learning_rate)**cumsum_loss[i])
            loss_log[t,i] = loss.calc_loss(outcomes[t],prediction[t,i], expert_wealth[i])
            expert_wealth[i] += outcomes[t] * (1 + prediction[t,i])
            wealth_log[t,i] =  expert_wealth[i]
            cumsum_loss[i] += loss_log[t,i]
    return learner_loss, loss_log, learner_prediction, weight_log,wealth_log

def weak_AA_class_wealth_ftl(outcomes, prediction, loss: LossFunc, Upper_Bound):
    T = len(outcomes)
    N_experts = np.size(prediction,1)
    norm_weights = np.ones(N_experts)
    norm_weights = norm_weights/ N_experts
    learner_prediction = np.zeros(T)
    loss_log = np.zeros((T, N_experts))
    cumsum_loss = np.zeros(N_experts)
    expert_wealth = np.ones(N_experts)
    expert_wealth = expert_wealth*1000000
    learner_wealth = 1000000
    learner_loss = np.zeros(T)
    weight_log = np.zeros((T, N_experts))
    wealth_log = np.zeros((T, N_experts))
    C = 1#(math.sqrt(np.log10(N_experts))) / Upper_Bound
    for t in range(T):
        #print(t)
        #if t % 10000 == 0:
         #   cumsum_loss = np.zeros(N_experts) 
        learner_prediction[t] = sum( norm_weights * prediction[t] )
        learner_loss[t] = loss.calc_loss(outcomes[t],learner_prediction[t], learner_wealth)
        learner_wealth += outcomes[t] * (1 + learner_prediction[t])
        denominator = 0
        learning_rate = C #/ math.sqrt(max(t,1))
        for j in range(N_experts):
                denominator +=  np.exp(learning_rate)**cumsum_loss[j]
        for i in range(N_experts):
            norm_weights[i] = (np.exp(learning_rate)**cumsum_loss[i]) / denominator
            weight_log[t,i] = (np.exp(learning_rate)**cumsum_loss[i])
            loss_log[t,i] = loss.calc_loss(outcomes[t],prediction[t,i], expert_wealth[i])
            expert_wealth[i] += outcomes[t] * (1 + prediction[t,i])
            wealth_log[t,i] =  expert_wealth[i]
            cumsum_loss[i] += loss_log[t,i]
    return learner_loss, loss_log, learner_prediction, weight_log,wealth_log


def weak_AA_class_wealth_pack(outcomes, prediction, loss: LossFunc, Upper_Bound):
    T = len(outcomes)
    N_experts = np.size(prediction,1)
    norm_weights = np.ones(N_experts)
    norm_weights = norm_weights/ N_experts
    learner_prediction = np.zeros(T)
    loss_log = np.zeros((T, N_experts))
    cumsum_loss = np.zeros(N_experts)
    expert_wealth = np.ones(N_experts)
    expert_wealth = expert_wealth*100000
    learner_wealth = 100000
    learner_loss = np.zeros(T)
    weight_log = np.zeros((T, N_experts))
    wealth_log = np.zeros((T, N_experts))
    C = (math.sqrt(np.log10(N_experts))) / Upper_Bound
    for t in range(T):
        #print(t)
        #if t % 10000 == 0:
         #   cumsum_loss = np.zeros(N_experts)
        learner_prediction[t] = sum( norm_weights * prediction[t] )
        learner_loss[t] = loss.calc_loss(outcomes[t],learner_prediction[t], learner_wealth)
        learner_wealth += outcomes[t] * (1 + learner_prediction[t])
        denominator = 0
        learning_rate = 1#C / math.sqrt(max(t,1))
        for j in range(N_experts):
                denominator +=  np.exp(learning_rate)**cumsum_loss[j]
        for i in range(N_experts):
            if t%200 == 0:
                norm_weights[i] = (np.exp(learning_rate)**cumsum_loss[i]) / denominator
            weight_log[t,i] = norm_weights[i]
            loss_log[t,i] = loss.calc_loss(outcomes[t],prediction[t,i], expert_wealth[i])
            expert_wealth[i] += outcomes[t] * (1 + prediction[t,i])
            wealth_log[t,i] =  expert_wealth[i]
            cumsum_loss[i] += loss_log[t,i]
    return learner_loss, loss_log, learner_prediction, weight_log,wealth_log

def weak_AA_class_drawdown(outcomes, prediction, loss: LossFunc, Upper_Bound,drawdown):
    T = len(outcomes)
    N_experts = np.size(prediction,1)
    norm_weights = np.ones(N_experts)
    learner_prediction = np.zeros(T)
    loss_log = np.zeros((T, N_experts))
    cumsum_loss = np.zeros(N_experts)
    learner_loss = np.zeros(T)
    C = (2 * math.sqrt(N_experts)) / Upper_Bound
    for t in range(1, T):
        #print(t)
        learner_prediction[t] = sum( norm_weights * prediction[t] )
        learner_loss[t] = loss.calc_loss(outcomes[t],learner_prediction[t],0)
        denominator = 0
        learning_rate = C / math.sqrt(t)
        for j in range(N_experts):
                denominator +=  np.exp(-learning_rate)**(cumsum_loss[j])
        for i in range(N_experts):
            norm_weights[i] = (np.exp(-learning_rate)**(cumsum_loss[i] )) / denominator
            loss_log[t,i] = loss.calc_loss(outcomes[t],prediction[t,i], drawdown[t, i])
            cumsum_loss += loss_log[t,:]
    return learner_loss, loss_log, learner_prediction

def weak_AA(outcomes, prediction, loss, learning_rate):
    T = len(outcomes)
    N_experts = np.size(prediction,1)
    norm_weights = np.ones(N_experts)
    learner_prediction = np.zeros(T)
    loss_log = np.zeros((T, N_experts))
    cumsum_loss = np.zeros(N_experts)
    learner_loss = np.zeros(T)
    for t in range(1, T):
        # print(t)
        learner_prediction[t] = sum( norm_weights * prediction[t] )
        learner_loss[t] = loss(outcomes[t],learner_prediction[t])
        for i in range(N_experts):
            denominator = 0
            for j in range(N_experts):
                denominator +=  np.exp(-learning_rate*cumsum_loss[j])
            norm_weights[i] = np.exp(-learning_rate*cumsum_loss[i]) / denominator
            loss_log[t,i] = loss(outcomes[t],prediction[t,i])
            cumsum_loss += loss_log[t,:]
    return learner_loss, loss_log, learner_prediction

def squash(ratio, S, offset):
    return offset + (1 / (1+ np.exp(-S *(ratio))))

#change
if __name__ ==  '__main__':
    data = pd.read_csv(r"C:\Users\Owner\Downloads\tennis1 (1).txt", sep='\s+')
    outs = data.iloc[:,1]
    p1 = data.iloc[:,3]
    p2 = data.iloc[:,5]
    p3 = data.iloc[:,7]
    p4 = data.iloc[:,9]
    pred = pd.concat([p1,p2, p3, p4], axis =1)
    
    lossfun = squaredLoss(outs.values, pred.values)
    AA_loss, experts_loss, pred_learner = AA_Class(outs.values, pred.values, lossfun, 2)
    WAA_loss, Wexperts_loss, Wpred_learner = weak_AA_class(outs.values, pred.values, lossfun, 2)
    
    