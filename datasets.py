import os 
import numpy as np


def PDGM(version='4vs2', flag=False): #version: 4vs3 or 4vs2
    X = []
    for i in range(1,11):
      fname = '/home/dixzhu/data/PDGM/PDGM_X'+str(i)+'.npz'
      tmp = np.load(fname)
      X.append(tmp['X'])
    X = np.concatenate(X, axis=0)
    fname = '/home/dixzhu/data/PDGM/PDGM_ids.npz'
    tmp = np.load(fname)
    pos_key_tr = {'4vs2':'tr_ids4','4vs3':'tr_ids4','GvsA':'tr_idsG','GvsO':'tr_idsG'}
    tr_ids_p = tmp[pos_key_tr[version]]
    te_ids_p = tmp[pos_key_tr[version].replace('tr','te')]
    neg_key_tr = {'4vs2':'tr_ids2','4vs3':'tr_ids3','GvsA':'tr_idsA','GvsO':'tr_idsO'}
    tr_ids_n = tmp[neg_key_tr[version]]
    te_ids_n = tmp[neg_key_tr[version].replace('tr','te')]
    tr_ids = np.concatenate([tr_ids_p,tr_ids_n], axis=0)
    te_ids = np.concatenate([te_ids_p,te_ids_n], axis=0)
    if flag == True:
      X = np.transpose(X, (0,1,4,2,3))
    else:
      X = np.transpose(X, (0,4,1,2,3))
    shuffle_ids = np.random.permutation(len(tr_ids))
    tr_ids = tr_ids[shuffle_ids]
    train_X = X[tr_ids]
    train_Y = np.concatenate([np.ones_like(tr_ids_p), np.zeros_like(tr_ids_n)], axis=0)
    train_Y = train_Y[shuffle_ids]
    train_Y = np.expand_dims(train_Y, axis=1)
    test_X = X[te_ids]
    test_Y = np.concatenate([np.ones_like(te_ids_p), np.zeros_like(te_ids_n)], axis=0)
    test_Y = np.expand_dims(test_Y, axis=1)
    print(train_X.shape)
    print(train_Y.shape)
    # print(train_Y)
    print(test_X.shape)
    print(test_Y.shape)
    # exit()
    return  (train_X, train_Y), (test_X, test_Y) 

if __name__ == '__main__':
    PDGM()
