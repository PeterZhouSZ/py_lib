#!/usr/bin/env python
"""
Classifier-related functions.

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2016-01
  modify  -  Feng Zhou (zhfe99@gmail.com), 2016-01
"""
import numpy as np


def parseSco(Sco, cTs, m=5):
  """
  Parse the score matrix and get the statistics:
    top-1 and top-5 hit
    rank of example

  Remark
    n: #examples
    k: #classes
    m: #top classes

  Input
    Sco   -  score matrix, n x k
    cTs   -  ground-truth label, n x
    m     -  #top classes, 1 | ... | {5} | ... | k

  Output
    C     -  top-m label, n x m
    co1   -  #top-1 hits
    co5   -  #top-m hits
    idx   -  rank of example (from worst to the best), n x
    ScoT  -  ground-truth score matrix, n x k
    D     -  confusion matrix, k x k
  """
  # dimension
  n, k = Sco.shape

  # top-5 label for each instance
  co1 = 0
  co5 = 0
  vis1 = np.zeros((n))
  vis5 = np.zeros((n))
  C = np.zeros((n, m))
  ScoT = np.zeros((n, k))
  D = np.zeros((k, k))

  # each object
  for i in xrange(n):
    # top-m label
    if k >= m:
      C[i][...] = Sco[i].argsort()[-1 : -m - 1 : -1]
    else:
      C[i][...] = Sco[i].argsort()[-1 : -2 : -1]

    # ground-truth label
    cT = cTs[i]
    ScoT[i, cT] = 1
    D[cT, C[i, 0]] += 1

    # check
    if C[i, 0] == cT:
      co1 += 1
      vis1[i] = 1
    if cT in C[i]:
      co5 += 1
      vis5[i] = 1

  # order
  dsts = np.sum((ScoT - Sco) ** 2, axis=1)
  vis1 = np.array(vis1) == 1
  vis5 = np.logical_and(np.array(vis5) == 1, np.logical_not(vis1))
  vis0 = np.logical_and(np.logical_not(vis1), np.logical_not(vis5))
  idx0 = np.argsort(dsts[vis0])[::-1]
  idx0 = np.nonzero(vis0)[0][idx0]
  idx5 = np.argsort(dsts[vis5])[::-1]
  idx5 = np.nonzero(vis5)[0][idx5]
  idx1 = np.argsort(dsts[vis1])[::-1]
  idx1 = np.nonzero(vis1)[0][idx1]

  idx = np.concatenate((idx0, idx5, idx1))

  return C, co1, co5, idx, ScoT, D


def mergeSco(Sco0, cT0s):
  """
  Merge the score from same images.

  Input
    Sco0  -  original score matrix, n0 x k
    cT0s  -  original ground-truth label, n0 x

  Output
    Sco   -  merged score matrix, n x k
    cTs   -  merged ground-truth label, n x
  """
  # dimension
  n0, k = Sco0.shape
  n = n0 / 2

  # map nm -> id
  nm2ids = self.dat.getNm2Ids()

  # matched pair
  vis = lib.zeros(n0)
  id1s, id2s = [], []
  for c in range(k):
    dct = nm2ids[c]
    nms = dct.keys()
    m = len(nms)
    for i in range(m):
      nmi = nms[i]
      idi = dct[nmi]
      if vis[idi] == 1:
        continue
      posi = nmi.rfind('_')
      if posi == -1:
        nmiCrop = lib.strDelSub(nmi)
      else:
        nmiCrop = nmi[:posi]

      for j in range(i + 1, m):
        nmj = nms[j]
        idj = dct[nmj]
        if vis[idj] == 1:
          continue
        posj = nmj.rfind('_')
        if posj == -1:
          nmjCrop = lib.strDelSub(nmj)
        else:
          nmjCrop = nmj[:posj]

        # check
        if nmiCrop == nmjCrop:
          id1s.append(idi)
          id2s.append(idj)
          vis[idi] = 1
          vis[idj] = 1
          break
  assert len(id1s) == n
  # assert(max(id1s) == n0 - 1)

  # merge score
  Sco = lib.zeros((n, k))
  cTs = lib.zeros(n)
  for i in range(n):
    id1, id2 = id1s[i], id2s[i]
    for c in range(k):
      Sco[i, c] = (Sco0[id1, c] + Sco0[id2, c]) / 2
    assert(cT0s[id1] == cT0s[id2])
    cTs[i] = cT0s[id1]

  return Sco, cTs
