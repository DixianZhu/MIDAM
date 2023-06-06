import torch
import numpy as np
from torch import Tensor
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized
from torch.utils.data.sampler import Sampler

class imbalanced_sampler(Sampler):
  r"""
    imbalanced sampling for +/- classes
  """
  #data_source: Sized
  #imratio: float
  #batch_size: int
  def __init__(self, data_source, imratio=0.1, batch_size=64, idx=-1, sample_scale=1.0, shuffle=True):
    self.data_source = data_source
    self.imratio = imratio
    self.shuffle = shuffle
    self.batch_size = batch_size
    self.num_samples = len(self.data_source)
    self.labels = data_source.get_labels()
    self.sample_scale = sample_scale
    ids = np.arange(self.num_samples)
    self.idx = idx
    if isinstance(idx,int) and idx == -1: # whole
      self.pos_pool = ids[self.labels == 1]
      self.neg_pool = ids[self.labels == 0]
    else: # subset
      self.num_samples = len(idx)
      self.pos_pool = np.intersect1d(ids[self.labels == 1], idx)
      self.neg_pool = np.intersect1d(ids[self.labels == 0], idx)
    self.pos_len = len(self.pos_pool)
    self.neg_len = len(self.neg_pool)
  def get_pos_len(self):
    return self.pos_len
  def get_neg_len(self):
    return self.neg_len
  def __iter__(self):
    num_samples = int(self.num_samples*self.sample_scale)
    if self.imratio == None: # uniform sampling
      if isinstance(self.idx,int) and self.idx == -1: # whole
        tmp = np.arange(self.num_samples)
        if self.shuffle: 
          np.random.shuffle(tmp)
        return iter(tmp[:num_samples]) 
      else: # subset
        if self.shuffle: 
          np.random.shuffle(self.idx)
        return iter(self.idx[:num_samples])
    pos_num = round(self.imratio * self.batch_size)
    neg_num = self.batch_size - pos_num
    sampled = []
    if self.shuffle: 
      np.random.shuffle(self.pos_pool)
      np.random.shuffle(self.neg_pool)
    pos_id = 0
    neg_id = 0
    for i in range(int(num_samples/self.batch_size)):
      for j in range(pos_num):
        sampled.append(self.pos_pool[pos_id % self.pos_len])
        pos_id += 1
      for j in range(neg_num):
        sampled.append(self.neg_pool[neg_id % self.neg_len])
        neg_id += 1
    return iter(sampled)

  def __len__(self):
    return self.num_samples
