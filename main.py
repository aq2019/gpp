import pandas as pd
import json
import numpy as np
import torch
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from solver import lgbmSolver, NNSolver
from utils import cross_validation, get_eln_param
from dataset import PriceDataset
from sklearn.model_selection import ParameterGrid

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


lgbm_hparams = {'boosting_type': ['dart'],
                'reg_lambda': [6.0, 8.0, 10.0],
                'objective': ['huber'],
                'verbose': [-1],
                'feature_fraction': [0.95], 
                #'bagging_fraction': [0.8],
                #'bagging_freq': [6],
                'num_leaves': [12, 15],  
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
    price_data.prepare_feature()
    price_data.save_features('features_final.csv', 'data_final.csv')
    #price_data.load_features('features_final.csv', 'data_final.csv')

    print("========="*4)
    if not np.all(np.isfinite(price_data.features.iloc[:, ~price_data.features.columns.isin(['Time', 'logreturn'])])) \
            or np.any(np.isnan(price_data.features.iloc[:, ~price_data.features.columns.isin(['Time', 'logreturn'])])):
        print("nan or inf in the data!!")
    else:
        #hparams = lgbm_hparams  
        hparams = nn_hparams
        #hparams = ridge_hparams
        
        best_r2_res = (-np.inf, np.inf, -np.inf)
        best_r2_conf = None
        best_mse_res = (-np.inf, np.inf, -np.inf)
        best_mse_conf = None
        best_corr_res = (-np.inf, np.inf, -np.inf)
        best_corr_conf = None
        count = 1
        for g in ParameterGrid(hparams):
            alg = NNSolver(897, 128 ,1) #1023
            #alg = lgbmSolver() 
            #alg =  Lasso() # Ridge() # 

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
nn
===r2=======r2=======r2=======r2=======r2====
(-0.007029189551073345, 0.00025018748684652926)
{'lr': 0.001, 'weight_decay': 2.1}
"""
""" 
Ridge
===r2=======r2=======r2=======r2=======r2====
(-0.021697069946226534, 0.00023975688257134995)
{'alpha': 1000000.0}
====mse=======mse=======mse=======mse=======mse===
(-0.021697069946226534, 0.00023975688257134995)
{'alpha': 1000000.0}
 """
""" 
 Lasso
===r2=======r2=======r2=======r2=======r2====
(-0.023438224777755767, 0.00024011452646721283)
{'alpha': 19000000.0}
====mse=======mse=======mse=======mse=======mse===
(-0.023438224777755767, 0.00024011452646721283)
{'alpha': 19000000.0}

 """
""" 
 dart
 ===r2=======r2=======r2=======r2=======r2====
(-0.009444106021066424, 0.00023643359037571642, 0.1321532830330821)
{'boosting_type': 'dart', 'early_stopping_rounds': 30, 'feature_fraction': 0.95, 'learning_rate': 0.005, 'num_boost_round': 150, 'num_leaves': 15, 'objective': 'huber', 'reg_lambda': 6.0, 'verbose': -1}
====mse=======mse=======mse=======mse=======mse===
(-0.010360091136239568, 0.0002362479174612726, 0.14370463872427272)
{'boosting_type': 'dart', 'early_stopping_rounds': 30, 'feature_fraction': 0.95, 'learning_rate': 0.005, 'num_boost_round': 150, 'num_leaves': 15, 'objective': 'huber', 'reg_lambda': 8.0, 'verbose': -1}
"""
""" 
===r2=======r2=======r2=======r2=======r2====
(-0.015151075107785017, 0.00023799627940427015, 0.11632750291661624)
{'boosting_type': 'gbdt', 'early_stopping_rounds': 30, 'feature_fraction': 0.9, 'learning_rate': 0.005, 'num_boost_round': 150, 'num_leaves': 12, 'objective': 'huber', 'reg_lambda': 6.0, 'verbose': -1}
====mse=======mse=======mse=======mse=======mse===
(-0.015151075107785017, 0.00023799627940427015, 0.11632750291661624)
{'boosting_type': 'gbdt', 'early_stopping_rounds': 30, 'feature_fraction': 0.9, 'learning_rate': 0.005, 'num_boost_round': 150, 'num_leaves': 12, 'objective': 'huber', 'reg_lambda': 6.0, 'verbose': -1}
"""