import lightgbm as lgb
from lightgbm import LGBMModel
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pricedataloader import TorchDataset
from model import PriceNet
from utils import to_tensor, to_numpy
from torch import nn
import math
import numpy as np


class Solver():
    def __init_(self):
        pass 

    def set_params(self, **params):
        pass
    
    def fit(self, x, y):
        pass
    
    def predict(self, x):
        pass

    def save_model(self, path):
        pass

    def load_model(self, path):
        pass


class lgbmSolver(Solver):
    def __init__(self):
        super().__init__()
        self.model = None
        self.params = None
    
    def set_params(self, **params):
        self.params = params

    def fit(self, x, y, val_ratio=0.2, **g):
        valid_size = int(len(y) * val_ratio)
        train_data = lgb.Dataset(x[: -valid_size], label=y[: -valid_size])
        _valid_data = lgb.Dataset(x[-valid_size:], label=y[-valid_size:])
        self.model = lgb.train(self.params, train_data, valid_sets=_valid_data)
    
    def predict(self, x):
        return self.model.predict(x)

    def save_model(self, path):
        self.model.booster_.save_model(path)

    def load_model(self, path):
        self.model = lgb.Booster(model_file = path)
    

class NNSolver(Solver):
    def __init__(self, input_dim, h_dim, output_dim):
        super().__init__()
        self.model = PriceNet(input_dim, h_dim, output_dim)
        if torch.cuda.is_available():
                self.model = self.model.cuda()
        self.loss_func = torch.nn.SmoothL1Loss(reduction='sum')
        self.weight_decay = 0.0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.writer_tr = SummaryWriter()
        self.writer_val = SummaryWriter()

    def set_params(self, **params):
        self.optimizer = torch.optim.Adam(self.model.parameters(), **params)

    def train_epoch(self, epoch, train_loader, optimizer, loss_func, log_interval = 5):
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            x, y = data[0], data[1].reshape(-1, 1)
            optimizer.zero_grad()
            ypred = self.model(x)
            loss = loss_func(ypred, y)
            loss.backward()
            train_loss += loss.item()
            self.writer_tr.add_scalar('Gradient/fc1', np.linalg.norm(self.model.fc1.weight.grad.cpu().data, 'fro'), batch_idx)
            self.writer_tr.add_scalar('Gradient/fc2', np.linalg.norm(self.model.fc3.weight.grad.cpu().data, 'fro'), batch_idx)
            self.optimizer.step()
            self.writer_tr.add_scalar('Loss/train', loss, batch_idx)
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:2.5f}'.format(
                    epoch, batch_idx * len(y), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(y)))
        ave_trainloss = train_loss / len(train_loader.dataset)
        print('====> Epoch: {} Average loss: {:2.5f}'.format(
              epoch, ave_trainloss))
        self.writer_tr.add_scalar('Loss/ave_train', ave_trainloss, epoch)
        
    def test_epoch(self, epoch, test_loader, loss_func):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data[0], data[1].reshape(-1, 1)
                ypred = self.model(x)
                loss = loss_func(y, ypred)
                test_loss += loss.item()
        ave_testloss = test_loss / len(test_loader.dataset)
        print('====> Test set loss: {:2.5f}'.format(ave_testloss))
        self.writer_val.add_scalar('Loss/valid', ave_testloss, epoch)
        return ave_testloss

    def _prepare_dataloader(self, x, y, batch_size, shuffle = True):
        x, y = x.to_numpy(), y.to_numpy()
        x, y = to_tensor(x), to_tensor(y)
        dataset = TorchDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader

    def fit(self, x, y, val_ratio=0.2, loss_func=None, optimizer=None, epochs = 200, batch_size = 64, **g):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                m.bias.data.uniform_(-0.01, 0.01)
        self.set_params(**g)

        valid_size = int(len(y) * val_ratio)
        x_tr, y_tr = x[: - valid_size], y[: - valid_size] 
        x_valid, y_valid =  x[-valid_size:], y[-valid_size:]
        tr_dataloader = self._prepare_dataloader(x_tr, y_tr, batch_size)
        valid_dataloader = self._prepare_dataloader(x_valid, y_valid, valid_size, 
                                                    shuffle=False)
        if loss_func is None:
            loss_func = self.loss_func
        if optimizer is None:
            optimizer = self.optimizer

        #for param_group in optimizer.param_groups:
        #    print(param_group['lr'], param_group['weight_decay'])

        best_test_loss = np.inf
        count = 0
        for epoch in range(1, epochs + 1):
            self.train_epoch(epoch, tr_dataloader, optimizer, loss_func)
            current_test_loss = self.test_epoch(epoch, valid_dataloader, loss_func)
            for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.99
            #if current_test_loss > best_test_loss:
            #    count += 1
            #    if count > 15:
            #        break
            if current_test_loss < best_test_loss:
                best_test_loss = current_test_loss
                count = 0
                self.save_model('nn_model.chp')
                
    
    def predict(self, x):
        #self.load_model('nn_model.chp')
        x = x.to_numpy()
        x_tensor = to_tensor(x)
        ypred = self.model(x_tensor)
        ypred = to_numpy(ypred)
        return ypred.reshape(-1)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

