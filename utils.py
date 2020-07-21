import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import shutil

def cross_validation(num_fold, alg, dataset, **g):
    res_r2 = np.zeros(num_fold)
    res_mse = np.zeros(num_fold)
    res_corr = np.zeros(num_fold)
    for idx in range(num_fold):
        tr, valid = dataset.cv_data(idx)
        y_train = tr['logreturn']
        x_train = tr.drop(['logreturn', 'Time'], axis=1)
        y_test = valid['logreturn']
        x_test = valid.drop(['logreturn', 'Time'], axis=1)
        alg.fit(x_train, y_train, **g) # **g
        #alg.save_model('./results/lgbm_'+str(idx)+'.lgbm')
        ypred = alg.predict(x_test)
        res_r2[idx] = r2_score(y_test, ypred)
        res_mse[idx] = mean_squared_error(y_test, ypred)
        res_corr[idx] =  np.corrcoef(y_test, ypred)[0, 1]
    return res_r2.mean(), res_mse.mean(), res_corr.mean()


def whiten(X, method='zca'):
    """
    Whitens the input matrix X using specified whitening method.
    Inputs:
        X:      Input data matrix with data examples along the first dimension
        method: Whitening method. Must be one of 'zca', 'zca_cor', 'pca',
                'pca_cor', or 'cholesky'.
    """
    X = X.reshape((-1, np.prod(X.shape[1:])))
    X_centered = X - np.mean(X, axis=0)
    Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    W = None
    
    if method in ['zca', 'pca', 'cholesky']:
        U, Lambda, _ = np.linalg.svd(Sigma)
        if method == 'zca':
            W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))
        elif method =='pca':
            W = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T)
        elif method == 'cholesky':
            W = np.linalg.cholesky(np.dot(U, np.dot(np.diag(1.0 / (Lambda + 1e-5)), U.T))).T
    elif method in ['zca_cor', 'pca_cor']:
        V_sqrt = np.diag(np.std(X, axis=0))
        P = np.dot(np.dot(np.linalg.inv(V_sqrt), Sigma), np.linalg.inv(V_sqrt))
        G, Theta, _ = np.linalg.svd(P)
        if method == 'zca_cor':
            W = np.dot(np.dot(G, np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T)), np.linalg.inv(V_sqrt))
        elif method == 'pca_cor':
            W = np.dot(np.dot(np.diag(1.0/np.sqrt(Theta + 1e-5)), G.T), np.linalg.inv(V_sqrt))
    else:
        raise Exception('Whitening method not found.')

    return np.dot(X_centered, W.T)

def to_tensor(x):
    _temp = torch.from_numpy(x)
    if torch.cuda.is_available():
        _temp = _temp.cuda()
    return _temp


def to_numpy(x):
    x = x.detach()
    if x.is_cuda:
        x = x.cpu()
    x = x.numpy()
    return x

def get_eln_param(g):
    alpha = g['a'] + g['b']
    l1_ratio = g['a'] / alpha
    qq = {}
    qq['alpha'] = alpha
    qq['l1_ratio'] =l1_ratio
    return qq