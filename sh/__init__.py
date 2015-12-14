import os
if os.getenv('has_display', '0') == '0':
  import matplotlib as mpl
  mpl.use('Agg')

from sh_com import *
from sh_img import *
from sh_hst import *
