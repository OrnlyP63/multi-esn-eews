import torch
import torch.nn as nn
from torch.optim import AdamW, Adam, SGD
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from scipy import sparse

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
torch.cuda.manual_seed(42)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def train_step(model:torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    ### Training
    train_loss, train_acc = 0, 0
    
    # Put model into training mode
    model.train()
    
    # Add a loop to through the training batches
    for batch, (X, y) in enumerate(data_loader):
        # Put data on target device
        X, y = X.to(device), y.to(device)
        
        #1. Forward pass
        y_pred = model(X)
        
        # 2. Calculate loss and accuracy (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss # accumulate train loss
        train_acc += accuracy_fn(y_true = y.argmax(dim = 1), y_pred = y_pred.argmax(dim = 1))
        # 3. Optimizer zero grad
        optimizer.zero_grad()
        
        # 4. Loss backward
        loss.backward()
        
        # 5. Optimizer step
        optimizer.step()
        
        
                    
    # Divide total train loss by length of train dataloader
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    
    # Print out what's happening
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%")
    

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    """Performs a testing loop step on model going over data_loader."""
    test_loss, test_acc = 0, 0
    
    # Put model in eval mode
    model.eval()
    
    # Turn on inference mode contest manager
    with torch.inference_mode():
        for X, y in data_loader:
            # Send the data to the target device
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate the loss/acc
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true = y.argmax(dim = 1), y_pred = test_pred.argmax(dim = 1))
            
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f} %\n")
        

class seq_learning(object):
    def __init__(self, params):
        self.params = list(params)
        self.beta = self.params[-1]
        self.beta.data.fill_(0)
        self.cuda = self.beta.is_cuda
        self.dim_label, self.dim_feature = self.beta.data.size()
        
    def init_train(self, H, Y_batch):
        batch_size = Y_batch.shape[0]
        Y_batch = Y_batch.float()
        H_t = H.t()
        P = torch.inverse(torch.matmul(H_t, H))
        Beta = torch.matmul(torch.matmul(P, H_t), Y_batch)
        
        self.beta.data = Beta.t().data
        
        return P, Beta
    
    def seq_train(self, H, Y_batch, P_current, beta_current):
        batch_size = Y_batch.shape[0]
        Y_batch = Y_batch.float()
        I = torch.eye(batch_size)
        
        if self.cuda:
            I = I.cuda()
            
        term1 = torch.matmul(P_current, H.t())
        term2 = torch.inverse(I + torch.matmul(H, torch.matmul(P_current, H.t())))
        term3 = torch.matmul(H, P_current)
        
        P_new = P_current - torch.matmul(term1, torch.matmul(term2, term3))
        
        term4 = torch.matmul(P_new, H.t())
        term5 = Y_batch - torch.matmul(H, beta_current)
        Beta_new = beta_current + torch.matmul(term4, term5)
        
        self.beta.data = Beta_new.t().data
        
        return P_new, Beta_new
    
class learning(object):
    # params is a list of parameters that need to be trained in this learning object
    # C is the regularization coefficient
    def __init__(self, params, C = 0.01):
        self.params = list(params)
        self.beta = self.params[-1] # assign the last parameter in the list to be self.beta
        self.cuda = self.beta.is_cuda # check if beta is on the GPU
        self.beta.data.fill_(0) # fill beta with zeros
        self.dim_label, self.dim_feature = self.beta.data.size() # get the dimensions of beta
        self.C = C # assign the regularization coefficient
        
    def init_train(self, H, Y):
        Y = Y.float() # convert Y to float
        H_t = H.t() # transpose H
        if self.cuda:
            I = torch.eye(H.shape[1]).cuda() # create an identity matrix with dimensions of H if on GPU
        else:
            I = torch.eye(H.shape[1]) # create an identity matrix with dimensions of H if on CPU

        HtH = torch.matmul(H_t, H) # multiply H_t and H together

        P = torch.inverse(HtH + self.C * I) # calculate the inverse of (HtH + C * I)
        Beta = torch.matmul(torch.matmul(P, H_t), Y) # calculate Beta using the formula Beta = P * H_t * Y
        
        self.beta.data = Beta.t().data # assign the transpose of Beta to the data of self.beta
        
        return P, Beta
    
class learning02(object):
    def __init__(self, params, C = 0.01):
        self.params = list(params)
        self.beta = self.params[-1]
        self.cuda = self.beta.is_cuda
        self.beta.data.fill_(0)
        self.cuda = self.beta.is_cuda
        self.dim_label, self.dim_feature = self.beta.data.size()
        self.C = C
        
    def init_train(self, H, Y):
        Y = Y.float()
        H_t = H.t()
        if self.cuda:
            I = torch.eye(H.shape[0]).cuda()
        else:
            I = torch.eye(H.shape[0])
        HHt = torch.matmul(H, H_t)

        P = torch.inverse(HHt + self.C * I)
        Beta = torch.matmul(H_t, torch.matmul(P, Y))
        
        self.beta.data = Beta.t().data
        
        return P, Beta
