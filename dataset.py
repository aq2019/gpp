import numpy as np
import pandas as pd
import ta 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils import whiten

ewm_alphas = [0.05]
def ewm_features(df):
    df_column_name = df.columns
    for i, alpha in enumerate(ewm_alphas):
        ewma = df[df_column_name].ewm(alpha=alpha).mean()
        if i == 0:
            df_n = (df - ewma) / ewma
        else:
            # df_normalized = df_normalized.join((df - ewma) / ewma, lsuffix='', rsuffix=str(alpha))
            # df = df.join((df[df_column_name] - ewma) / ewma, lsuffix='', rsuffix='_ema_n' + str(alpha))
            # df = (df[df_column_name] - ewma) / ewma
            df_n = df_n.join((df - ewma) / ewma, lsuffix='', rsuffix='ewma')
        del ewma
    # print("Joining results!")
    # df = df.join(df_normalized, lsuffix='', rsuffix='_ema')
    df_n = df_n.replace([np.inf, -np.inf], np.nan)
    df_n = df_n.fillna(0.0)
    return df_n

class PriceDataset():
    def __init__(self, num_fold = 5, test_ratio = 0.05):
        self.num_fold = num_fold
        self.test_ratio = test_ratio
    
        self.price = None
        self.features = None
        self.aux_data = {}
        self.data_final = None

        self.start_date = None
        self.end_date = None
        self.datasize = 0
        self.trainsize = 0

        self.target_date_feature = None
        self.transform = StandardScaler()
        self.PCA_transform = PCA(n_components = 1083, whiten=True)
    
    def save_features(self, fea_path, data_path, target_path):
        self.data_final.to_csv(data_path, index = False)
        self.features.to_csv(fea_path, index = False)
        self.target_date_feature.to_csv(target_path, index = False)

    def load_features(self, fea_path, data_path, target_path):
        self.target_date_feature = pd.read_csv(target_path)
        self.data_final = pd.read_csv(data_path)
        self.features = pd.read_csv(fea_path)
        self.datasize = len(self.data_final)
        self.start_date = self.data_final.iloc[0, 0]
        self.end_date = self.data_final.iloc[self.datasize - 1, 0]
        self.trainsize = self.datasize - int(self.test_ratio * self.datasize)

    def read_price_data(self, path):
        _temp = pd.read_csv(path)
        _temp = _temp.drop(['Change'], axis=1)
        self.price = _temp[::-1][1:].copy(deep = True)

    def add_auxdata(self, name, path):
        if name in self.aux_data:
            print("Data already exists")
            return
        _temp = pd.read_csv(path)
        _temp = _temp.drop(['Change'], axis=1)
        self.aux_data[name] = _temp[::-1][1:].copy(deep = True)
    
    def prepare_feature(self):
        _temp = PriceDataset._feature_engineering(self.price)
        _temp['logreturn'] = np.log(_temp['Last'].shift(-4)/_temp['Last']) 
        #_temp['logreturn'] = _temp['Last'].shift(-4) - _temp['Last']      
        _temp = _temp.drop(['Open', 'High', 'Low', 'Last'], axis = 1)
        df = _temp.copy(deep=True)

        for symb in self.aux_data:
            if symb[-2:] == '_v':
                _temp = PriceDataset._feature_engineering(self.aux_data[symb])
                if symb[:-2] == "^btcusd":
                    _temp = _temp.drop(["trend_mass_index", "volatility_kcp"], axis = 1)
            else:
                _temp = PriceDataset._feature_engineering(self.aux_data[symb], volume=False)
            _temp = _temp.drop(['Open', 'High', 'Low', 'Last'], axis = 1)
            df = pd.merge(df, _temp, on='Time', suffixes=('', '_'+symb))

        self.data_final = df[112:].copy(deep=True)  #remove the first 112 and the last 4 where there is no 'y'
        self.data_final = self.data_final.fillna(method='bfill')

        tt = self.data_final.loc[:, ~self.data_final.columns.isin(['Time', 'logreturn'])].copy(deep=True)
        #tt = ewm_features(tt)# .clip(-20.0, 20.0)
        tt = pd.DataFrame(self.transform.fit_transform(tt))
        #tt = self.PCA_transform.fit_transform(self.data_final.loc[:, ~self.data_final.columns.isin(['Time', 'logreturn'])])
        #tt = pd.DataFrame(whiten(tt.values))
        self.target_date_feature = tt.iloc[-1, :].copy(deep=True)
        
        self.features = tt[1:-4].copy(deep=True)
        self.features['Time'] = self.data_final['Time'].values[1:-4]
        self.features['logreturn'] = self.data_final['logreturn'].values[1:-4]

        self.datasize = len(self.data_final)
        self.start_date = self.data_final.iloc[0, 0]
        self.end_date = self.data_final.iloc[self.datasize - 1, 0]
        self.trainsize = self.datasize - int(self.test_ratio * self.datasize)
    
    @staticmethod
    def _feature_engineering(df, volume=True):
        if volume:
            df = ta.add_all_ta_features(df, open="Open", high="High", low="Low",
                                        close="Last", volume="Volume", fillna=False)
        else:
            df = ta.add_momentum_ta(df, high="High", low="Low",
                                    close="Last", volume="Volume", fillna=False)
            df = ta.add_volatility_ta(df, high="High", low="Low",
                                    close="Last", fillna=False)
            df = ta.add_trend_ta(df, high="High", low="Low",
                                    close="Last", fillna=False)
            df = ta.add_others_ta(df, close="Last", fillna=False)
        df["trend_psar_up"] = df["trend_psar_up"].fillna(0.0)
        df["trend_psar_down"] = df["trend_psar_down"].fillna(0.0)
        return df

    def cv_data(self, fold):
        if fold > self.num_fold:
            print("fold larger than num_fold")
            return
        if fold < 0:
            print('fold number should be greater than 0')
            return
        
        _piece_size = int(self.trainsize / (2 * self.num_fold - 1))
        train_df = self.features[fold*_piece_size : (fold+self.num_fold-1)*_piece_size - 5]
        if fold == self.num_fold - 1:
            valid_df = self.features[(fold+self.num_fold-1)*_piece_size : self.trainsize]
        else:
            valid_df = self.features[(fold+self.num_fold-1)*_piece_size : (fold+self.num_fold)*_piece_size]
        return train_df, valid_df

    def get_test_data(self):
        test_df = self.features[self.trainsize + 5 :]
        return test_df


