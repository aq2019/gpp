import pandas as pd
import json
import numpy as np
import torch
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from solver import lgbmSolver, NNSolver
from utils import cross_validation, get_eln_param
from dataset import PriceDataset
from sklearn.model_selection import ParameterGrid

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

path = "./datacopy/"
gold_symbol = "^xauusd"
symbol_lst_vol = ["^btcusd", "gld", "shv",
                  "iei", "shy","ief", "tlt"] #gold etf and bond etf 
symbol_lst = ["^xagusd", "^xptusd", "dxy", #indices
              "vix", "spx"] #gold etf and bond etf "dowi", "nasx", 
np.set_printoptions(precision=4)
torch.set_default_dtype(torch.float64)
num_fold = 5

nn_hparams = { 'weight_decay': [0.5 * u for u in range(5, 15, 2)],
            'lr': [1e-2*u for u in range(1, 3)]}

ridge_hparams = {'alpha': [1e6*u for u in range(1, 20, 1)]}


lgbm_hparams = {'boosting_type': ['gbdt', 'dart'],
                'reg_lambda': [4.0, 6.0, 8.0, 10.0],
                'objective': ['huber'],
                'verbose': [-1],
                'feature_fraction': [0.9, 0.95], 
                #'bagging_fraction': [0.8],
                #'bagging_freq': [6],
                'num_leaves': [8, 12, 16],  
                'learning_rate': [0.005, 0.001],
                'early_stopping_rounds': [30],  
                'num_boost_round': [150],  
                }
 

if __name__ == '__main__':
    price_data = PriceDataset(num_fold)
    price_data.read_price_data(path+gold_symbol+".csv")
    for symb in symbol_lst:
        price_data.add_auxdata(symb, path+symb+".csv")
    for symb in symbol_lst_vol:
        price_data.add_auxdata(symb+'_v', path+symb+".csv")
    #price_data.prepare_feature()
    #price_data.save_features('features_final_ewma.csv', 'data_final_ewma.csv', 'target_final_ewma.csv')
    price_data.load_features('features_final_ewma.csv', 'data_final_ewma.csv', 'target_final_ewma.csv',)

    print("========="*4)
    if not np.all(np.isfinite(price_data.features.iloc[:, ~price_data.features.columns.isin(['Time', 'logreturn'])])) \
            or np.any(np.isnan(price_data.features.iloc[:, ~price_data.features.columns.isin(['Time', 'logreturn'])])):
        print("nan or inf in the data!!")
    else:
        #hparams = lgbm_hparams  
        #hparams = nn_hparams
        hparams = ridge_hparams
        
        best_r2_res = (-np.inf, np.inf, -np.inf)
        best_r2_conf = None
        best_mse_res = (-np.inf, np.inf, -np.inf)
        best_mse_conf = None
        best_corr_res = (-np.inf, np.inf, -np.inf)
        best_corr_conf = None
        count = 1
        for g in ParameterGrid(hparams):
            #alg = NNSolver(897, 64 ,1) #1023
            #alg = lgbmSolver() 
            alg =  Ridge() # Lasso() 

            print("************"*5)
            print(count)
            count += 1
            print("************"*5)
            print(g)
            
            alg.set_params(**g)
            
            r2, mse, corr = cross_validation(num_fold, alg, price_data, **g)
            
            if best_r2_res[0] <= r2:
                best_r2_res = (r2, mse, corr)
                best_r2_conf = g
            if best_mse_res[1] >= mse:
                best_mse_res = (r2, mse, corr)
                best_mse_conf = g
            if best_corr_res[2] <= corr:
                best_corr_res = (r2, mse, corr)
                best_corr_conf = g
            print(r2, mse, corr)
        
        print("===r2===="*5)
        print(best_r2_res)
        print(best_r2_conf)
        print("====mse==="*5)
        print(best_mse_res)
        print(best_mse_conf)
        print("====corr==="*5)
        print(best_corr_res)
        print(best_corr_conf)


""" 
final
===r2=======r2=======r2=======r2=======r2====
(-0.005333120651336798, 0.0002342893098355686, 0.1975090919486559)
{'boosting_type': 'dart', 'early_stopping_rounds': 30, 'feature_fraction': 0.95, 'learning_rate': 0.005, 'num_boost_round': 150, 'num_leaves': 16, 'objective': 'huber', 'reg_lambda': 8.0, 'verbose': -1}
====mse=======mse=======mse=======mse=======mse===
(-0.005333120651336798, 0.0002342893098355686, 0.1975090919486559)
{'boosting_type': 'dart', 'early_stopping_rounds': 30, 'feature_fraction': 0.95, 'learning_rate': 0.005, 'num_boost_round': 150, 'num_leaves': 16, 'objective': 'huber', 'reg_lambda': 8.0, 'verbose': -1}
final_ewma
===r2=======r2=======r2=======r2=======r2====
(-0.014286672955967994, 0.0002371502478032773, 0.10988593884169126)
{'boosting_type': 'dart', 'early_stopping_rounds': 30, 'feature_fraction': 0.95, 'learning_rate': 0.005, 'num_boost_round': 150, 'num_leaves': 8, 'objective': 'huber', 'reg_lambda': 10.0, 'verbose': -1}
====mse=======mse=======mse=======mse=======mse===
(-0.014286672955967994, 0.0002371502478032773, 0.10988593884169126)
{'boosting_type': 'dart', 'early_stopping_rounds': 30, 'feature_fraction': 0.95, 'learning_rate': 0.005, 'num_boost_round': 150, 'num_leaves': 8, 'objective': 'huber', 'reg_lambda': 10.0, 'verbose': -1}
"""