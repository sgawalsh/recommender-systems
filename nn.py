# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 16:57:30 2021

@author: sgawalsh
"""
import math
import numpy as np
import torch
import data
import itertools
from sklearn.model_selection import train_test_split
import copy
from torch.optim.lr_scheduler import _LRScheduler
import pickle

class my_net(torch.nn.Module):
    def __init__(self, n_users, n_items, u_embeds, i_embeds, hidden_sizes = [500, 500, 500], drop_rates = [.5, 0.5, .25], embed_drop = .05):
        super(my_net, self).__init__()
        self.user_embeds = torch.nn.Embedding(n_users, u_embeds)
        self.item_embeds = torch.nn.Embedding(n_items, i_embeds)
        
        def make_hidden(n_in): # generator to create hidden layers
                
                nonlocal hidden_sizes, drop_rates
                assert len(drop_rates) <= len(hidden_sizes)
                
                for n_out, rate in itertools.zip_longest(hidden_sizes, drop_rates):
                    yield torch.nn.Linear(n_in, n_out)
                    yield torch.nn.ReLU()
                    if rate is not None and rate > 0.:
                        yield torch.nn.Dropout(rate)
                    n_in = n_out 

        self.embed_drop = torch.nn.Dropout(embed_drop)
        self.hidden = torch.nn.Sequential(*list(make_hidden(u_embeds + i_embeds)))
        self.out_layer = torch.nn.Linear(hidden_sizes[-1], 1)
        self._init()


    def forward(self, user, item):
        x = torch.cat([self.user_embeds(user), self.item_embeds(item)], dim=1)
        x = self.embed_drop(x)
        x = self.hidden(x)
        x = self.out_layer(x)
        x = torch.sigmoid(x)

        return x
    
    def _init(self):
        """
        Setup embeddings and hidden layers with reasonable initial values.
        """
        
        def init(m):
            if type(m) == torch.nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
                
        weight = .01
        self.user_embeds.weight.data.uniform_(-1 *weight, weight)
        self.item_embeds.weight.data.uniform_(-1 * weight, weight)
        self.hidden.apply(init)
        init(self.out_layer)
    
class batch_iterator:
    def __init__(self, x, y, shuffle = True, bs = 32):
        if shuffle:
            p = np.random.permutation(x.shape[0])
            self.x, self.y = x[p], y[p]
        else:
            self.x, self.y = x, y
            
        self.max_batch = x.shape[0] // bs
        self.bc = 0
        self.bs = bs
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.bc < self.max_batch:
            bc = self.bc
            self.bc += 1
            return self.x[bc * self.bs: (bc + 1) * self.bs], self.y[bc * self.bs: (bc + 1) * self.bs]
        else:
            raise StopIteration()
            
class CyclicLR(_LRScheduler):
    
    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]



def cosine(t_max, eta_min=0):
    
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min)*(1 + math.cos(math.pi*t/t_max))/2
    
    return scheduler

def train_model(norm = 'mm'):

    ratings = data.get_data("ml-latest-small/ratings.csv").to_numpy()
    
    y = ratings[~np.isnan(ratings)]
    
    # ratings = (ratings - y.mean()) / y.std()
    # y = (y - y.mean()) / y.std()
    
    if norm == 'mm': # minmax scaling
        y = (y - y.min()) / (y.max() - y.min())
    elif norm == 'z': # z scaling
        y = (y - y.mean()) / y.std()
    
    x = np.column_stack(np.nan_to_num(ratings).nonzero())
    
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=1)
    
    datasets = {
        "train": (x_train, y_train),
        "val": (x_valid, y_valid)
        }
    
    
    n_users, n_items = ratings.shape
    
    u_embeds, i_embeds = 150, 150
    
    lr = 5e-4
    wd = 1e-1
    bs = 2000
    n_epochs = 100
    patience = 10
    no_improvements = 0
    best_loss = np.inf
    best_weights = None
    history = []
    
    # use GPU if available
    identifier = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(identifier)
    
    base = 500
    net = my_net(n_users, n_items, u_embeds, i_embeds, [base, base, base], [.5, .5, .25])
    net.to(device)
    print(net)
    criterion = torch.nn.MSELoss(reduction='sum')
    #criterion = torch.nn.L1Loss(reduction='sum')
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    iterations_per_epoch = int(math.ceil(datasets['train'][0].shape[0] // bs))
    scheduler = CyclicLR(optimizer, cosine(t_max=iterations_per_epoch * 2, eta_min=lr/10))
    lr_history = []
    
    fmt = '[{epoch:03d}/{total:03d}] train: {train:.4f} - val: {val:.4f}'
    
    # start training
    for epoch in range(n_epochs):
        stats = {'epoch': epoch + 1, 'total': n_epochs}
        
        for phase in ('train', 'val'):
            if phase == 'train':
                net.train()
            else:
                net.eval()
            training = phase == 'train'
            running_loss = 0.0
            
            itr = batch_iterator(*datasets[phase], bs = bs)
            
            for x_b, y_b in itr:
                optimizer.zero_grad()
                x_b_t = torch.LongTensor(x_b).to(device)
                y_b_t = torch.FloatTensor(y_b).view(-1, 1).to(device)
                preds = net(x_b_t[:, 0], x_b_t[:, 1])
                loss = criterion(preds, y_b_t)
                if training:
                    scheduler.step()
                    loss.backward()
                    optimizer.step()
                    lr_history.extend(scheduler.get_lr())
                
                        
                running_loss += loss.item()
                
            epoch_loss = running_loss / datasets[phase][0].shape[0]
            stats[phase] = epoch_loss
            if phase == 'val':
                if epoch_loss < best_loss:
                    print('loss improvement on epoch: %d' % (epoch + 1))
                    best_loss = epoch_loss
                    best_weights = copy.deepcopy(net.state_dict())
                    no_improvements = 0
                else:
                    no_improvements += 1
                    
        history.append(stats)
        print(fmt.format(**stats))
    
        if no_improvements >= patience:
            break
        
    net.load_state_dict(best_weights)
    preds = net(torch.empty(n_items, dtype = torch.long).fill_(1).to(device), torch.arange(n_items).to(device))
        
    torch.save(best_weights, "models/best_weights.pt")
    pickle.dump(net.user_embeds, open("models/user_embeds", "wb"))
    pickle.dump(net.item_embeds, open("models/item_embeds", "wb"))
    
def get_predictions(u_id = 1, n = 10, verbose = True):
    
    identifier = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(identifier)
    
    ratings = data.get_data("ml-latest-small/ratings.csv").to_numpy()
    n_users, n_items = ratings.shape
    u_embeds, i_embeds = 150, 150
    
    model = my_net(n_users, n_items, u_embeds, i_embeds)
    model.to(device)
    model.load_state_dict(torch.load("models/best_weights.pt"))
    model.eval()
    
    preds = model(torch.empty(n_items, dtype = torch.long).fill_(u_id).to(device), torch.arange(n_items).to(device))
    
    recs = preds.cpu().detach().numpy().flatten()
    recs = np.column_stack((np.arange(n_items), recs))
    recs = recs[np.isnan(ratings[u_id])] # remove previously rated user item interactions
    recs = recs[recs[:,1].argsort()]
    recs = np.flipud(recs)
    if n:
        recs = recs[:n]
        
    if verbose:
        movies = data.get_data_raw("ml-latest-small/movies.csv").to_numpy()
        print("Reccomendations for user {}:\n".format(u_id))
        for i, m in enumerate(recs):
            print("{}. {} - {}".format(i + 1, movies[int(m[0])][1], str(round(m[1], 4))))
            
    return recs
    
        
#train_model('mm')
#get_predictions(10, 0)
    