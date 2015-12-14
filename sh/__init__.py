import os
if os.getenv('has_display', '0') == '0':
  import matplotlib as mpl
  mpl.use('Agg')
  import pdb; pdb.set_trace()

from sh_com import *
from sh_img import *
from sh_hst import *
