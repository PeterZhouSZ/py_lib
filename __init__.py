#!/usr/bin/env python
"""
Init py_lib module.

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2015-09
  modify  -  Feng Zhou (zhfe99@gmail.com), 2016-01
"""
import os
from sh import *
from pri import *
from util import *
from img import *
from fio import *
from cell import *
from str import *
from dct import *
from dst import *
from tran import *
from lib_clf import *

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
