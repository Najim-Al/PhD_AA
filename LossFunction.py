# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 15:31:01 2021

@author: Owner
"""
import abc
import numpy as np
from AA_coursework import *
class LossFunc(metaclass=abc.ABCMeta):
    
    
    @abc.abstractmethod
    def calc_loss()->float:
        pass


class LongShortLoss(LossFunc):

    def calc_loss(self, outcome, prediction)->float:
        N_assets = 1
        return_to_portfolio = 0
        for n in range(N_assets):
            return_to_portfolio += outcome * prediction
        loss = -np.log(1 + return_to_portfolio)
        return loss
   
class squaredLoss(LossFunc):

    def calc_loss(self, outcome, prediction)->float:
        loss = (prediction - outcome)**2
        return loss


class CombinedLoss(LossFunc):
    def __init__(self,**kwargs):

        # check for optional args
        
        #get values
        self.return_scale = kwargs['return_scale']
        self.ls = kwargs['ls']
        self.dls = kwargs['dls']
    '''    
    def calc_loss(self)->float:
        N_assets = 1
        return_to_portfolio = 0
        for n in range(N_assets):
            return_to_portfolio +=  ((self.ls / (self.ls +self. dls)) * 
                                     (self.outcome * self.prediction))  + (
                                         (self.dls / (self.ls + self.dls)) *
                                         min(self.outcome * self.prediction, 0))
        loss = -np.log(1 + self.return_scale * return_to_portfolio)
        return loss
    '''
    def calc_loss(self,outcome, prediction)->float:
        N_assets = 1
        return_to_portfolio = 0
        for n in range(N_assets):
            return_to_portfolio +=  ((self.ls / (self.ls +self. dls)) * 
                                     (outcome * prediction))  + (
                                         (self.dls / (self.ls + self.dls)) * 
                                         min((outcome * prediction) , 0))
        loss = -np.log(1 + self.return_scale * return_to_portfolio)
        #if loss == 0:
         #   loss = 1
        return loss
 
def squash(ratio, S, offset):
    return offset + (1 / (1+ np.exp(-S *(ratio))))
    
class PnLLoss(LossFunc):
    def __init__(self,**kwargs):

        # check for optional args
        
        #get values
        self.return_scale = kwargs['return_scale']
        self.cof = kwargs['cof']
    '''    
    def calc_loss(self)->float:
        N_assets = 1
        return_to_portfolio = 0
        for n in range(N_assets):
            return_to_portfolio +=  ((self.ls / (self.ls +self. dls)) * 
                                     (self.outcome * self.prediction))  + (
                                         (self.dls / (self.ls + self.dls)) *
                                         min(self.outcome * self.prediction, 0))
        loss = -np.log(1 + self.return_scale * return_to_portfolio)
        return loss
    '''
    def calc_loss(self,outcome, prediction)->float:
        N_assets = 1
        return_to_portfolio = 0
        for n in range(N_assets):
            return_to_portfolio +=   squash((outcome + (outcome * prediction)) / (outcome + 1),self.cof, 0.5)  
        loss = -np.log( self.return_scale * return_to_portfolio)
        #if loss == 0:
         #   loss = 1
        return loss
    
class PnL_weak_loss(LossFunc):
    def __init__(self,**kwargs):

        # check for optional args
        
        #get values
        self.return_scale = kwargs['return_scale']
    '''    
    def calc_loss(self)->float:
        N_assets = 1
        return_to_portfolio = 0
        for n in range(N_assets):
            return_to_portfolio +=  ((self.ls / (self.ls +self. dls)) * 
                                     (self.outcome * self.prediction))  + (
                                         (self.dls / (self.ls + self.dls)) *
                                         min(self.outcome * self.prediction, 0))
        loss = -np.log(1 + self.return_scale * return_to_portfolio)
        return loss
    '''
    def calc_loss(self,outcome, prediction)->float:
        N_assets = 1
        return_to_portfolio = 0
        for n in range(N_assets):
            #return_to_portfolio +=   -1 * (prediction * outcome)
            return_to_portfolio +=   -1 * (outcome + (prediction * outcome))
            #return_to_portfolio +=   -1 * min((outcome + (prediction * outcome)),0)
        loss =  self.return_scale * return_to_portfolio
        #if loss == 0:
         #   loss = 1
        return loss
    
class PnL_weak_loss_dd(LossFunc):
    def __init__(self,**kwargs):

        # check for optional args
        
        #get values
        self.return_scale = kwargs['return_scale']
    '''    
    def calc_loss(self)->float:
        N_assets = 1
        return_to_portfolio = 0
        for n in range(N_assets):
            return_to_portfolio +=  ((self.ls / (self.ls +self. dls)) * 
                                     (self.outcome * self.prediction))  + (
                                         (self.dls / (self.ls + self.dls)) *
                                         min(self.outcome * self.prediction, 0))
        loss = -np.log(1 + self.return_scale * return_to_portfolio)
        return loss
    '''
    def calc_loss(self,outcome, prediction, drawdown)->float:
        N_assets = 1
        return_to_portfolio = 0
        for n in range(N_assets):
            #return_to_portfolio +=   -1 * (prediction * outcome)
            return_to_portfolio +=   -1 * ( (outcome + (prediction * outcome))) #+ 0.25* drawdown)
            #return_to_portfolio +=   -1 * min((outcome + (prediction * outcome)),0)
        loss =  self.return_scale * return_to_portfolio
        #if loss == 0:
         #   loss = 1
        return loss