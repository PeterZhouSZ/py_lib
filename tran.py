#!/usr/bin/env python
"""
Transformation-related functions.

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2015-12
  modify  -  Feng Zhou (zhfe99@gmail.com), 2015-12
"""

def tranPt(para, Pt0, h, w, tran='scale2'):
  """
  Transform a set of 2D point.

  Input
    para  -  parameter, 2 x | 3 x | 4 x | 5 x
               tran == 'scale2': para = [dx, dy]
    Pt0   -  original position, n x 2
    h     -  height
    w     -  width
    tran  -  transformation type, {'scale2'} | 'scale' | 'rigid'

  Output
    pt    -  new position, 2 x
  """
  # dimension
  n = len(Pt0)

  # each point
  Pt = []
  for i in range(n):
    y0, x0 = Pt0[i]

    # normalize x0, y0
    x0 = 2 * (1.0 * x0 / (w - 1)) - 1
    y0 = 2 * (1.0 * y0 / (h - 1)) - 1

    # transform
    if tran == 'scale2':
      y = y0 * 0.5 + para[1]
      x = x0 * 0.5 + para[0]

    else:
      raise Exception('unknown tran: {}'.format(tran))

    # back to original coordinates
    x = (w - 1) * (x + 1) / 2
    y = (h - 1) * (y + 1) / 2
    Pt.append([int(y), int(x)])

  return Pt
