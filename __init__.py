#!/usr/bin/env python
"""
Init.

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2015-09
  modify  -  Feng Zhou (zhfe99@gmail.com), 2015-11
"""
import os
if os.getenv('has_display', '0') == '1':
  from sh import *
from pri import *
from util import *
from img import *
from fio import *
from cell import *
from match import *
from str import *
from dct import *
from dst import *

def init(prL=2, isOut=False):
  """
  Init.

  Input
    prL    -  prompt level, {2} | 3 | ...
    isOut  -  output mode, True | {False}
  """
  prSet(prL)
  np.seterr(under='ignore')
  if os.getenv('has_display', '0') == '1':
    shIni(isOut=isOut)
