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
        #N_assets = 1
        return_to_portfolio = 0
        return_to_portfolio +=  ((self.ls / (self.ls +self. dls)) * 
                                  (outcome * (1+prediction)))  + (
                                      (self.dls / (self.ls + self.dls)) * 
                                      min((outcome * (1+prediction)) , 0))
         #loss = -np.log(1 + (self.return_scale * (return_to_portfolio + outcome)))
        #if loss == 0:
         #   loss = 1
        return_to_portfolio =  self.return_scale*return_to_portfolio
        #return_to_portfolio +=  (self.return_scale*(outcome*(1+prediction)))
        # if return_to_portfolio < -1:
        #     return_to_portfolio = -0.99
        loss = -1*return_to_portfolio
        return loss
    
class CombinedLosshedge(LossFunc):
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
        #N_assets = 1
        return_to_portfolio = 0
        return_to_portfolio +=  self.return_scale * ( (self.ls / (self.ls +self. dls) ) * 
                                  (outcome * (1+prediction)) )  + (
                                      (self.dls / (self.ls + self.dls)) * 
                                      min((outcome * (1+prediction)) , 0))
        # loss = -np.log(1 + (self.return_scale * (return_to_portfolio + outcome)))
        #if loss == 0:
         #   loss = 1
         
        #return_to_portfolio +=  (self.return_scale*(outcome*(1+prediction)))
        # if abs(return_to_portfolio) > 1:
        #     return_to_portfolio = 0
        loss = -np.log(1+return_to_portfolio)
        return loss
 
    
class CombinedLossMulti(LossFunc):
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
                                     sum((outcome * prediction)))  + (
                                         (self.dls / (self.ls + self.dls)) * 
                                         min(sum(outcome * prediction) , 0))
        loss = -np.log(1+ self.return_scale * return_to_portfolio)
        #if loss == 0:
         #   loss = 1
        return loss
    
    
    
class LS_pnl_loss(LossFunc):
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
        loss = -(self.return_scale * return_to_portfolio)
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
    
class PnL_weak_loss_wealth(LossFunc):
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
    def calc_loss(self,outcome, prediction, wealth)->float:
        # N_assets = 1
        return_to_portfolio = 0
        # for n in range(N_assets):
            #return_to_portfolio +=    (-1*prediction *outcome) / max(abs(outcome),1)
            #return_to_portfolio +=     -( np.sqrt(abs(prediction))) * outcome
            #return_to_portfolio +=     ( (prediction * outcome) + outcome)
            # if outcome != 0:    
        return_to_portfolio +=     ((wealth + (self.return_scale *(outcome) * (1+prediction) )) ) / wealth
        if return_to_portfolio < 0:
            return_to_portfolio = 0.01
        #     else:
        #         return_to_portfolio = 1
        #     #return_to_portfolio +=   #-1*max((outcome + (prediction * outcome)),0)
        # if return_to_portfolio > 1:
        #     return_to_portfolio = 1
        # elif return_to_portfolio < -1:
        #     return_to_portfolio = -1
        loss =    np.log10(return_to_portfolio)
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
        # N_assets = 1
        return_to_portfolio = 0
        # for n in range(N_assets):
            #return_to_portfolio +=    (-1*prediction *outcome) / max(abs(outcome),1)
            #return_to_portfolio +=     -( np.sqrt(abs(prediction))) * outcome
            #return_to_portfolio +=     ( (prediction * outcome) + outcome)
            # if outcome != 0:    
        return_to_portfolio +=     (outcome*(1+prediction))*self.return_scale
        
        #     else:
        #         return_to_portfolio = 1
        #     #return_to_portfolio +=   #-1*max((outcome + (prediction * outcome)),0)
        # if return_to_portfolio > 1:
        #     return_to_portfolio = 1
        # elif return_to_portfolio < -1:
        #     return_to_portfolio = -1
        loss =    -1*return_to_portfolio
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
            #return_to_portfolio +=   (prediction * outcome)
            return_to_portfolio +=    -drawdown#-(prediction * outcome) + 0.1*drawdown
            #return_to_portfolio +=   -1 * min((outcome + (prediction * outcome)),0)
        loss =  self.return_scale * return_to_portfolio
        #if loss == 0:
         #   loss = 1
        return loss