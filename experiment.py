from losses import AUCMLoss, CrossEntropyLoss, MIDAM_softmax_pooling_loss, MIDAM_attention_pooling_loss
from midam import MIDAM
from adam import Adam
from pesg import PESG
from models import ResNet20, ResNet20_stoc_MIL, ResNet20_MIL, FFNN, FFNN_MIL, FFNN_stoc_MIL, FFNN_softmax
from imbalanced_sampler import imbalanced_sampler
from utils import set_all_seeds, collate_fn, TabularDataset, evaluate_auc, MIL_sampling
from dataset import BreastCancer

import torch 
import numpy as np
import time
from sklearn.model_selection import KFold
import argparse

parser = argparse.ArgumentParser(description = 'MIDAM experiments')
parser.add_argument('--loss', default='MIDAM-att', type=str, help='loss functions to use.')
parser.add_argument('--dataset', default='MUSK1', type=str, help='the name for the dataset to use.')
parser.add_argument('--activation', default='sigmoid', type=str, help='activation function for the output layer for AUC loss.')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.1, type=float, help='momentum parameter for model update')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay for training the model')
parser.add_argument('--tau', default=0.1, type=float, help='hardness parameter for smoothed-max pooling')
parser.add_argument('--margin', default=1.0, type=float, help='margin parameter for AUC loss')
parser.add_argument('--SPR', default=0.5, type=float, help='sampling positive rate')
parser.add_argument('--batch_size', default=16, type=int, help='bag batch size per iteration')
parser.add_argument('--instance_batch_size', default=4, type=int, help='instance batch size per bag')
parser.add_argument('--seed', default=123, type=int, help='random seed for experiments')


args = parser.parse_args()
BATCH_SIZE = args.batch_size
lr = args.lr
weight_decay = args.decay
set_all_seeds(args.seed)
num_class = 1

if args.dataset in ['BreastCancer']:
  if args.loss in ['CELoss', 'AUCMLoss']:
    (train_data, train_labels), (test_data, test_labels) = BreastCancer(flag=True) # 3D data
  else:
    (train_data, train_labels), (test_data, test_labels) = BreastCancer(flag=False) # 2D data in bag
elif args.dataset in ['MUSK1','MUSK2','Fox','Tiger','Elephant']: 
  if args.dataset == 'MUSK1': 
    tmp = np.load('./data/musk_1.npz',allow_pickle=True)
  elif args.dataset == 'MUSK2': 
    tmp = np.load('./data/musk_2.npz',allow_pickle=True)
  elif args.dataset == 'Fox': 
    tmp = np.load('./data/fox.npz',allow_pickle=True)
  elif args.dataset == 'Tiger': 
    tmp = np.load('./data/tiger.npz',allow_pickle=True)
  elif args.dataset == 'Elephant': 
    tmp = np.load('./data/elephant.npz',allow_pickle=True)
  train_data = tmp['train_X']
  test_data = tmp['test_X']
  train_labels = tmp['train_Y'].astype(int)
  test_labels = tmp['test_Y'].astype(int)
  if args.seed != 123: # different random seed, we use different data splitting
    teN = len(test_labels)
    trN = len(train_labels)
    randIds = np.random.permutation(trN+teN)
    X = np.concatenate([train_data,test_data],axis=0)
    Y = np.concatenate([train_labels,test_labels],axis=0)
    X = X[randIds]
    Y = Y[randIds]
    train_data = X[:trN]
    test_data = X[trN:]
    train_labels = Y[:trN]
    test_labels = Y[trN:]


traindSet = TabularDataset(train_data, train_labels) # we don't use augmentation (even for image datasets).
testSet = TabularDataset(test_data, test_labels)
collate_function = collate_fn
DIMS_dict={'MUSK1':166, 'MUSK2':166, 'Fox':230, 'Tiger':230, 'Elephant':230, 'PDGM':155, 'BreastCancer': 672, 'Colon':256, 'hypertension': 31}
DIMS = DIMS_dict[args.dataset]
if args.dataset in ['PDGM','hypertension']:
  inchannels = 1
  if args.loss in ['CELoss','AUCMLoss']: # 3D model
    inchannels = DIMS
elif args.dataset in ['BreastCancer','Colon']:
  inchannels = 3
  

kf = KFold(n_splits=5)
N = len(traindSet)
tmpX = np.zeros((N,1))
if args.activation == 'sigmoid' or args.activation == 'l2' or args.activation == 'l1' or args.activation == 'scale':
   parameter_set = [0.1, 0.5, 1.0]
else:
  parameter_set = [0.1, 1, 10]

if args.loss in ['CELoss','CEmax','CEatt','CEmean','CEsoftmax']: # cross entropy loss don't use activation for AUC loss.
  parameter_set = [0] # no extra parameter tuned for CE type loss
  args.activation = None

testloader =  torch.utils.data.DataLoader(testSet, batch_size=1, num_workers=1, shuffle=False, collate_fn=collate_fn)


part = 0
print ('Start Training')
print ('-'*30)
for train_id, val_id in kf.split(tmpX):
  for para in parameter_set:
    trainloader =  torch.utils.data.DataLoader(dataset=traindSet, sampler=imbalanced_sampler(data_source=traindSet, batch_size=BATCH_SIZE, imratio=args.SPR, idx=train_id), batch_size=BATCH_SIZE, num_workers=1, shuffle=False, collate_fn=collate_fn)
    validloader =  torch.utils.data.DataLoader(dataset=traindSet, sampler=imbalanced_sampler(data_source=traindSet, batch_size=BATCH_SIZE,imratio=0.5,idx=val_id,sample_scale=1.0), batch_size=1, num_workers=1, shuffle=False, collate_fn=collate_fn)
    if args.dataset in ['MUSK1','MUSK2','Fox','Tiger','Elephant']:
      if args.loss in ['CEatt','AUCMatt']:
        model = FFNN_MIL(num_classes=num_class, last_activation=args.activation, dims=DIMS)
        print('Attention-based MIL-FFNN model')
      elif args.loss in ['MIDAM-att']:
        model = FFNN_stoc_MIL(num_classes=num_class, last_activation=None, dims=DIMS)
        print('Stochastic-attention-based MIL-FFNN model')
      elif args.loss in ['AUCMsoftmax', 'CEsoftmax']:
        model = FFNN_softmax(num_classes=num_class, last_activation=args.activation, dims=DIMS, tau=args.tau)
        print('Softmax-based FFNN model')
      else:
        model = FFNN(num_classes=num_class, last_activation=args.activation, dims=DIMS)
      model = model.cuda()
    else:
      if args.loss in ['CEatt','AUCMatt']:
        model = ResNet20_MIL(num_classes=1, last_activation=args.activation, inchannels=inchannels)
        print('Attention-based MIL-ResNet20 model')
      elif args.loss in ['MIDAM-att']:
        model = ResNet20_stoc_att(num_classes=1, last_activation=None, inchannels=inchannels)
        print('Stochastic-attention-based MIL-ResNet20 model')
      elif args.loss in ['AUCMsoftmax', 'CEsoftmax']:
        model = ResNet20_softmax(num_classes=1, last_activation=args.activation, inchannels=inchannels, tau=args.tau)
        print('Softmax-based ResNet20 model')
      else:
        model = ResNet20(num_classes=1, last_activation=args.activation, inchannels=inchannels)
      model = model.cuda()
      
    # define loss & optimizer
    if args.loss in ['AUCMLoss', 'AUCMmean', 'AUCMmax', 'AUCMatt', 'AUCMsoftmax']:
      Loss = AUCMLoss(margin=para)
    elif args.loss in ['MIDAM-att']:
      Loss = MIDAM_attention_pooling_loss(data_len=N, margin=para)
    elif args.loss in ['MIDAM-smx']:
      Loss = MIDAM_softmax_pooling_loss(data_len=N, margin=para, tau=args.tau)
    elif args.loss in ['CELoss','CEmax','CEatt','CEmean','CEsoftmax']:
      Loss = CrossEntropyLoss()
    if args.loss in ['AUCMLoss', 'AUCMmean', 'AUCMmax', 'AUCMatt', 'AUCMsoftmax']:
      optimizer = PESG(model, 
                       a=Loss.a, 
                       b=Loss.b, 
                       alpha=Loss.alpha, 
                       imratio=args.SPR, 
                       lr=lr,
                       epoch_decay=0, 
                       margin=para, 
                       weight_decay=weight_decay)
    elif args.loss in ['MIDAM-att','MIDAM-smx']:
      optimizer = MIDAM(model, a=Loss.a, b=Loss.b, lr=lr, weight_decay=weight_decay, momentum=args.momentum)  
    else:
      optimizer = Adam(model, lr=lr, weight_decay=weight_decay)  
    print('Margin=%s, part=%s'%(para, part))
    for epoch in range(100):
      tr_loss = 0
      if epoch in [50,75]:
          optimizer.update_lr(decay_factor=10)
          if args.loss in ['MIDAM-att','MIDAM-smx']:
            Loss.update_smoothing(decay_factor=2)
      start_time = time.process_time()
      for idx, data in enumerate(trainloader):
          train_data_bags, train_labels, ids = data
          train_data = []
          y_pred = []
          sd = []
          for i in range(len(ids)):
            if args.loss in ['CELoss', 'CEmean', 'CEatt', 'AUCMLoss', 'AUCMmean', 'AUCMatt', 'AUCMsoftmax', 'CEsoftmax']:
              tmp_pred = MIL_sampling(train_data_bags[i], model, batch_size=args.instance_batch_size, mode='plain') # when model is attention, it is auto-reduce
              y_pred.append(tmp_pred)
            elif args.loss in ['AUCMmax','CEmax']:
              tmp_pred = MIL_sampling(train_data_bags[i], model, batch_size=args.instance_batch_size, mode='max') # when model is attention, it is auto-reduce
              y_pred.append(tmp_pred)
            elif args.loss in ['MIDAM-smx']:
              tmp_pred = MIL_sampling(train_data_bags[i], model, batch_size=args.instance_batch_size, mode='exp') # when model is attention, it is auto-reduce
              y_pred.append(tmp_pred)
            elif args.loss in ['MIDAM-att']:
              tmp_pred, tmp_sd = MIL_sampling(train_data_bags[i], model, batch_size=args.instance_batch_size, mode='att') # when model is attention, it is auto-reduce
              y_pred.append(tmp_pred)
              sd.append(tmp_sd)
          y_pred = torch.cat(y_pred, dim=0) 
          if args.loss in ['MIDAM-att']:
            sd = torch.cat(sd, dim=0)
          ids = torch.from_numpy(np.array(ids)).cuda()
          train_labels = torch.from_numpy(np.array(train_labels)).cuda()
          if args.loss in [ 'MIDAM-att']:
            loss = Loss(y_pred=(y_pred, sd), y_true=train_labels, ids=ids)
          elif args.loss in [ 'MIDAM-smx']:
            loss = Loss(y_pred=y_pred, y_true=train_labels, ids=ids)
          else:
            loss = Loss(y_pred, train_labels).mean()
          tr_loss = tr_loss + loss.cpu().detach().numpy()
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      end_time = time.process_time()
      dur_time = end_time - start_time
      tr_loss = tr_loss/idx
      model.eval()
      print ('Epoch=%s, BatchID=%s, training_loss=%.4f, lr=%.4f'%(epoch, idx, tr_loss,  optimizer.lr))
      print ('Epoch=%s, BatchID=%s, time=%.4f'%(epoch, idx, dur_time))
      if args.loss in ['AUCMLoss', 'AUCMmean', 'AUCMmax', 'AUCMatt', 'MIDAM-att', 'MIDAM-smx', 'AUCMsoftmax']:
        print (str(Loss.a.cpu().detach().numpy())+', '+str(Loss.b.cpu().detach().numpy())+ ', '+str(Loss.alpha.cpu().detach().numpy()))
      tr_loss = 0
      with torch.no_grad():
        if args.loss in ['CEloss', 'CEmean', 'AUCMLoss', 'AUCMmean', 'AUCMatt', 'CEatt', 'CEsoftmax', 'AUCMsoftmax']:
          single_tr_auc = evaluate_auc(trainloader, model, mode='mean') 
          single_te_auc = evaluate_auc(testloader, model, mode='mean') 
          single_val_auc = evaluate_auc(validloader, model, mode='mean') 
        elif args.loss in ['MIDAM-att']:
          single_tr_auc = evaluate_auc(trainloader, model, mode='att') 
          single_te_auc = evaluate_auc(testloader, model, mode='att') 
          single_val_auc = evaluate_auc(validloader, model, mode='att') 
        elif args.loss in ['MIDAM-smx']:
          single_tr_auc = evaluate_auc(trainloader, model, mode='softmax', tau=args.tau) 
          single_te_auc = evaluate_auc(testloader, model, mode='softmax', tau=args.tau) 
          single_val_auc = evaluate_auc(validloader, model, mode='softmax', tau=args.tau) 
        else:
          single_tr_auc = evaluate_auc(trainloader, model, mode='max') 
          single_te_auc = evaluate_auc(testloader, model, mode='max') 
          single_val_auc = evaluate_auc(validloader, model, mode='max') 

        model.train()
        print ('Epoch=%s, BatchID=%s, Tr_AUC=%.4f, Val_AUC=%.4f, Test_AUC=%.4f, lr=%.4f'%(epoch, idx, single_tr_auc, single_val_auc, single_te_auc,  optimizer.lr))
  part += 1 


