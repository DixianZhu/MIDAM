import torch 
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc




def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def collate_fn(list_items):
     x = []
     y = []
     ids = []
     for x_, y_, ids_ in list_items:
         # print(f'x_={x_}, y_={y_}')
         x.append(x_)
         y.append(y_)
         ids.append(ids_)
     return x, y, ids

class TabularDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], int(idx)

    def get_labels(self):
        return np.concatenate(self.Y,axis=0)



def MIL_sampling(list_X, model, batch_size=1, mode='plain', tau=0.1):
    if type(list_X) == list:
      X = torch.from_numpy(np.concatenate(list_X, axis=0)).cuda()
    else: # it is a tensor
      X = torch.from_numpy(list_X).cuda()
    bag_size = X.shape[0]
    weights = torch.ones(bag_size)
    sample_size = min(bag_size, batch_size)
    ids = torch.multinomial(weights, sample_size, replacement=False)
    X = X[ids,...]
    if mode=='plain':
      y_pred_bag = model(X.float())
      y_pred = torch.mean(y_pred_bag.view([1,-1]), dim=1, keepdim=True)
      return y_pred
    elif mode=='exp':
      y_pred_bag = torch.exp(model(X.float())/tau)
      y_pred = torch.mean(y_pred_bag.view([1,-1]), dim=1, keepdim=True)
      return y_pred
    elif mode=='att':
      y_pred_bag, weights_bag = model(X.float())
      sn_bag = y_pred_bag * weights_bag
      sn = torch.mean(sn_bag.view([1,-1]), dim=1, keepdim=True)
      sd = torch.mean(weights_bag.view([1,-1]), dim=1, keepdim=True)
      return sn, sd
      
def argmax_offline(X, model):
    N = X.shape[0]
    iterN = int(np.ceil(N/16))
    maxvalue = -1e10
    for i in range(iterN):
      a = i*16
      b = min((i+1)*16, N)
      tmpX = X[a:b,...]
      y_pred_bag = model(tmpX.float())
    for j in range(b-a):
        if y_pred_bag[j] > maxvalue:
          maxvalue = y_pred_bag[j]
          maxid = j+a
    return maxid

def MIL_aggregation(list_X, model, idx=None, mode='max', tau=0.1, offline=False):
    # currently handle single data point.
    # can hanlde both list and tensor data.
    if idx==None:
      if type(list_X) == list:
        X = torch.from_numpy(np.concatenate(list_X, axis=0)).cuda()
      else: # it is a tensor
        X = torch.from_numpy(list_X).cuda()
      if offline==False:
        y_pred_bag = model(X.float())
      if mode=='max':
        if offline==False:
          y_pred = torch.max(y_pred_bag.view([1,-1]), dim=1, keepdim=True).values
        else:
          maxid = argmax_offline(X, model)
          y_pred = model(torch.unsqueeze(X[maxid,...].float(), dim=0))
      elif mode=='mean':
        y_pred = torch.mean(y_pred_bag.view([1,-1]), dim=1, keepdim=True)
      elif mode=='softmax':
        y_pred = tau*torch.log(torch.mean(torch.exp(y_pred_bag.view([1,-1])/tau), dim=1, keepdim=True))
      elif mode=='att':
        # y_pred = y_pred_bag
        y_pred = torch.sum(y_pred_bag[0].view([1,-1]) * torch.nn.functional.normalize(y_pred_bag[1].view([1,-1]),p=1.0,dim=-1), dim=1, keepdim=True)
      
    else:
      if type(list_X) == list:
        X = torch.from_numpy(list_X[idx]).cuda()
      else: # it is a tensor
        X = list_X[idx].cuda()
        X = torch.unsqueeze(X,dim=0)
      y_pred = model(X.float())
    
    return y_pred


def evaluate_auc(dataloader, model, mode='max', tau=0.1, debug=False):
  test_pred = []
  test_true = []
  for jdx, data in enumerate(dataloader):
    if True: # list data
      test_data_bags, test_labels, ids = data
      y_pred = []
      for i in range(len(ids)):
        if mode == 'max':
          tmp_pred = MIL_aggregation(test_data_bags[i],model,mode=mode,tau=tau,offline=True)
        else:
          tmp_pred = MIL_aggregation(test_data_bags[i],model,mode=mode,tau=tau,offline=False)
        y_pred.append(tmp_pred)
      y_pred = torch.cat(y_pred, dim=0)
    test_pred.append(y_pred.cpu().detach().numpy())
    test_true.append(test_labels)
  # print(test_true)
  test_true = np.concatenate(test_true, axis=0)
  test_pred = np.concatenate(test_pred, axis=0)
  if debug==True:
    tmp=np.concatenate([test_true,test_pred],axis=1)
    print(tmp[:50])
  single_te_auc =  roc_auc_score(test_true, test_pred) 
  return single_te_auc
