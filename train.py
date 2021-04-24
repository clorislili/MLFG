import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from model.cnn import CNN_1d_underfit
from model.rnn import RNN_1d
import timeit

def run_one_epoch(train_flag, dataloader, cnn_1d, optimizer, device="cuda"):

    torch.set_grad_enabled(train_flag)
    cnn_1d.train() if train_flag else cnn_1d.eval() 

    losses = []
    accuracies = []

    for (x,y) in dataloader: # collection of tuples with iterator

        (x, y) = ( x.to(device), y.to(device) ) # transfer data to GPU

        output = cnn_1d(x) # forward pass
        # print(type(output))
        # print(output.shape)
        if type(output) == tuple:
          output = output[0]
          output = torch.mean(output, dim=1)
          print('111111111')

        output = output.squeeze() # remove spurious channel dimension
        print(output.shape)
        print(y.shape)
        assert(0)
        loss = F.binary_cross_entropy_with_logits( output.unsqueeze(1), y ) # numerically stable

        if train_flag: 
            loss.backward() # back propagation
            optimizer.step()
            optimizer.zero_grad()

        losses.append(loss.detach().cpu().numpy())
        accuracy = torch.mean( ( (output > .5) == (y > .5) ).float() )
        accuracies.append(accuracy.detach().cpu().numpy())  
    
    return( np.mean(losses), np.mean(accuracies) )

def train_model(model, train_data, validation_data, epochs=100, patience=10, verbose = True, early = True, rnn=False, reverse=False):
    """
    Train a 1D CNN model and record accuracy metrics.
    """
    # Move the model to the GPU here to make it runs there, and set "device" as above

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 1. Make new BedPeakDataset and DataLoader objects for both training and validation data.

    
    # if reverse == True:
    #   # print('rrrrrrrrr')
    #   train_dataset = TensorDataset(train_data)
    #   train_loader = DataLoader(train_dataset, batch_size= 100)
    #   validation_dataset = TensorDataset(validation_data)
    #   validation_dataloader = DataLoader(train_dataset, batch_size= 100)
    # else:
      
    #   train_dataset = BedPeaksDataset(train_data, genome, cnn_1d.seq_len)
    #   train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, num_workers = 0)
    #   validation_dataset = BedPeaksDataset(validation_data, genome, cnn_1d.seq_len)
    #   validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=1000)
    train_dataloader = train_data
    validation_dataloader = validation_data
    # 2. Instantiates an optimizer for the model. 

    optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)

    # 3. Run the training loop with early stopping. 

    train_accs = []
    
    val_accs = []
    patience_counter = patience
    best_val_loss = np.inf
    check_point_filename = 'cnn_1d_checkpoint_lxq_327_rnn.pt'
    for epoch in range(100):
        start_time = timeit.default_timer()
        if rnn == True:
          train_loss, train_acc = run_one_epoch(True, train_dataloader, model, optimizer, device, rnn)
          val_loss, val_acc = run_one_epoch(False, validation_dataloader, model, optimizer, device, rnn)
        else:
          train_loss, train_acc = run_one_epoch(True, train_dataloader, model, optimizer, device)
          val_loss, val_acc = run_one_epoch(False, validation_dataloader, model, optimizer, device)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        if early == True:
          if val_loss < best_val_loss: 
              torch.save(model.state_dict(), check_point_filename)
              best_val_loss = val_loss
              patience_counter = patience
          else: 
              patience_counter -= 1
              if patience_counter <= 0: 
                  model.load_state_dict(torch.load(check_point_filename)) # recover the best model so far
                  break
        elapsed = float(timeit.default_timer() - start_time)
        print("Epoch %i took %.2fs. Train loss: %.4f acc: %.4f. Val loss: %.4f acc: %.4f. Patience left: %i. Best_val_loss: %.4f" % 
              (epoch+1, elapsed, train_loss, train_acc, val_loss, val_acc, patience_counter, best_val_loss))

    # 4. Return the fitted model (not strictly necessary since this happens "in place"), train and validation accuracies.

    return model, train_accs, val_accs


x = pd.read_csv("/home/lxq/train_x.csv")
# x = x.drop(columns=['Unnamed: 0','sample_name','range_end','chrm', 'end'])
x = x.drop(columns=['Unnamed: 0','sample_name'])
x = x.to_numpy()
x = torch.from_numpy(x).type(torch.FloatTensor)
train_size = int(len(x) * 0.8)
train_x = x[:train_size]
test_x = x[train_size:]
y = pd.read_csv("/home/lxq/train_y.csv")
y = y.drop(columns=['Unnamed: 0','sample_name'])
#y['1-y'] = y['y'].apply(lambda x: 1 - x)
y = y.to_numpy()
y = torch.from_numpy(y).type(torch.FloatTensor)
train_y = y[:train_size]
test_y = y[train_size:]
train_dataset = TensorDataset(train_x[0:10000].unsqueeze(2), train_y[0:10000])
test_dataset = TensorDataset(test_x[0:1000].unsqueeze(2), test_y[0:1000])
train_loader = DataLoader(train_dataset, batch_size= 100)
test_loader = DataLoader(test_dataset, batch_size= 100)


#model = CNN_1d_underfit()
model = RNN_1d()
print(model)

my_rnn1d, train_accs, val_accs = train_model(model, train_loader, test_loader)