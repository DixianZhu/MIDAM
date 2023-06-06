import os 
import numpy as np



def Colon(flag=False): #version: 4vs3 or 4vs2
    fname = './data/colon.npz'
    tmp = np.load(fname)
    Y = tmp['Y']
    if flag == True:
      X = tmp['oriX']
      X = np.expand_dims(X,axis=1)
    else:
      X = tmp['X']
    X = np.transpose(X,[0,1,4,2,3])
    N = Y.shape[0]
    ids = np.random.permutation(N)
    trN = int(0.9 * N)
    tr_ids = ids[:trN]
    te_ids = ids[trN:]
    train_X = X[tr_ids]
    test_X = X[te_ids]
    train_Y = Y[tr_ids]
    test_Y = Y[te_ids]
    print(train_X.shape)
    print(train_Y.shape)
    print(test_X.shape)
    print(test_Y.shape)
    return  (train_X, train_Y), (test_X, test_Y) 



def BreastCancer(flag=False): #version: 4vs3 or 4vs2
    fname = './data/breast.npz'
    tmp = np.load(fname)
    Y = tmp['Y']
    if flag == True:
      X = tmp['oriX']
      X = np.expand_dims(X,axis=1)
    else:
      X = tmp['X']
    X = np.transpose(X,[0,1,4,2,3])
    N = Y.shape[0]
    ids = np.random.permutation(N)
    trN = int(0.9 * N)
    tr_ids = ids[:trN]
    te_ids = ids[trN:]
    train_X = X[tr_ids]
    test_X = X[te_ids]
    train_Y = Y[tr_ids]
    test_Y = Y[te_ids]
    print(train_X.shape)
    print(train_Y.shape)
    print(test_X.shape)
    print(test_Y.shape)
    return  (train_X, train_Y), (test_X, test_Y) 


def hypertension(flag=False): #version: 4vs3 or 4vs2
    fname = '/dual_data/not_backed_up/hypertension-OCT/bscans.npy'
    X = np.load(fname)
    fname = '/dual_data/not_backed_up/hypertension-OCT/ground_truth.npy'
    Y = np.load(fname)
    if flag == True:
      X = np.transpose(X, (0,1,2,3,4))
    else:
      X = np.transpose(X, (0,2,1,3,4))
    Y = np.expand_dims(Y, axis=1)
    N = Y.shape[0]
    ids = np.random.permutation(N)
    trN = int(0.9*N)
    tr_ids = ids[:trN]
    te_ids = ids[trN:]
    train_X = X[tr_ids]
    train_Y = Y[tr_ids]
    test_X = X[te_ids]
    test_Y = Y[te_ids]
    print(train_X.shape)
    print(train_Y.shape)
    print(test_X.shape)
    print(test_Y.shape)
    return  (train_X, train_Y), (test_X, test_Y) 


def PDGM(version='4vs2', flag=False): #version: 4vs3 or 4vs2
    X = []
    for i in range(1,11):
      fname = './data/PDGM/PDGM_X'+str(i)+'.npz'
      tmp = np.load(fname)
      X.append(tmp['X'])
    X = np.concatenate(X, axis=0)
    fname = './data/PDGM/PDGM_ids.npz'
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
    print(test_X.shape)
    print(test_Y.shape)
    return  (train_X, train_Y), (test_X, test_Y) 

if __name__ == '__main__':
    PDGM()
