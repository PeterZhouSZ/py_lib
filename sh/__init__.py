#!/usr/bin/env python
"""
Init py_lib.sh modules.

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2015-09
  modify  -  Feng Zhou (zhfe99@gmail.com), 2015-12
"""

import os
if os.getenv('has_display', '0') == '0':
  import matplotlib as mpl
  mpl.use('Agg')

from sh_com import *
from sh_img import *
from sh_hst import *
