#!/usr/bin/env python
"""
File IO utility functions.

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2015-03
  modify  -  Feng Zhou (zhfe99@gmail.com), 2016-01
"""
import os
import csv


def loadLns(inpath):
  """
  Read a file and return its content as list of lines.

  Input
    inpath  -  input path, string

  Output
    lines   -  lines, 1 x n (list)
  """
  fio = open(inpath, 'r')
  lines = fio.read().splitlines()
  fio.close()

  return lines


def saveLns(outpath, lines, subx='\n'):
  """
  Write a list of line into a file.

  Input
    outpath  -  output path, string
    lines    -  lines, 1 x n (list)
    subx     -  subfix of each line, {None} | '\n' | ...
  """
  assert outpath.__class__ == str
  fio = open(outpath, 'w')
  for line in lines:
    try:
      fio.write(line)
    except UnicodeEncodeError:
      fio.write(line.encode('utf8'))

    if subx is not None:
      fio.write(subx)
  fio.close()


def mkDir(dirPath, mkL=0):
  """
  Make a fold if not existed.

  Input
    dirPath  -  directory path
    mkL      -  operation level if fold already exists, {0} | 1
                  0: do nothing
                  1: del the fold
  """
  if dirPath == '':
    return

  if not os.path.exists(dirPath):
    os.makedirs(dirPath)

  else:
    if mkL == 1:
      import shutil
      shutil.rmtree(dirPath, ignore_errors=True)
      os.makedirs(dirPath)


def cpFile(pathSrc, pathDst, svL=1):
  """
  Copy file.

  Input
    pathSrc  -  src path
    pathDst  -  dst path
    svL      -  save level, {1} | 2
                  1: write to pathDst even it exist
                  2: not write to pathDst if it exist
  """
  import shutil
  if os.path.exists(pathDst) and svL == 1:
    return
  shutil.copyfile(pathSrc, pathDst)


def rmFile(path):
  """
  Delete file if exist.

  Input
    path  -  src path
  """
  if os.path.exists(path):
    os.remove(path)


def save(filepath, data, svL=1):
  """
  Save data as a pickle-format file.

  Input
    filepath  -  file name
    data      -  data
    svL       -  save level, 0 | {1} | 2
                   0: write to pathDst even it exist
                   1: write to pathDst even it exist
                   2: not write to pathDst if it exist
  """
  import cPickle

  if svL == 0 or filepath is None:
    return

  # create fold if not exist
  foldPath = os.path.dirname(filepath)
  mkDir(foldPath)

  with open(filepath, "w") as fo:
    cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)


def load(filename):
  """
  Load data from a pickle-format file.

  Input
    filename  -  filename

  Output
    data      -  data
  """
  import cPickle

  with open(filename, 'r') as fo:
    data = cPickle.load(fo)
  return data


def loadH5(filename, varNm, dtype=None):
  """
  Load data from a hdf5 file.

  Input
    filename  -  filename
    varNm     -  variable name
    dtype     -  type, {None} | np.double | ...

  Output
    data      -  data
  """
  import h5py
  import numpy as np

  file = h5py.File(filename, 'r')
  data0 = file[varNm]
  if dtype is None:
    data = np.asarray(data0)
  else:
    data = np.asarray(data0, dtype=dtype)
  file.close()
  return data


def saveH5(filename, data, varNm):
  """
  Save data in hdf5 file.

  Input
    filename  -  filename
    data      -  data
    varNm     -  variable name
  """
  import h5py
  file = h5py.File(filename, "w")
  file.create_dataset(varNm, data=data)
  file.close()


def savePath(fold, prex, subx=None, type=None):
  """
  Get the save path.

  Input
    fold  -  fold
    prex  -  path prefix
    subx  -  subfix, {None}
    type  -  type, None

  Output
    path  -  file path
  """
  saveFold = os.getenv('save', os.path.join(os.environ['HOME'], 'save'))
  saveFold = os.path.join(saveFold, fold)

  # create fold if necessary
  mkDir(saveFold)

  # subfix
  if subx is not None:
    prex = prex + '_' + subx

  if type is None:
    path = os.path.join(saveFold, prex)
  elif type == 'txt':
    path = os.path.join(saveFold, prex + '.txt')
  else:
    raise Exception('unknown type: {}'.format(type))
  return path


def exist(nm, type='file'):
  """
  Check whether the name eixst.

  Input
    nm    -  name
    type  -  type, {'file'}

  Output
    res   -  status, 'True' | 'False'
  """
  if nm is None:
    return False

  if type == 'file':
    return os.path.isfile(nm)

  else:
    raise Exception('unknown type: {}'.format(type))


def listFold(fold):
  """
  Return the list of all folders under a folder.

  Input
    fold       -  root fold

  Output
    foldNms    -  directory name list, 1 x n (list)
    foldPaths  -  directory path list, 1 x n (list)
  """
  foldNms = []
  foldPaths = []

  for foldNm in os.listdir(fold):
    # fold absolute path
    foldPath = os.path.join(fold, foldNm)

    # skip non fold
    if not os.path.isdir(foldPath):
      continue

    # store
    foldNms.append(foldNm)
    foldPaths.append(foldPath)

  return foldNms, foldPaths


def listFoldR(fold):
  """
  Return the list of all folders recursively under a folder.

  Input
    fold       -  root fold

  Output
    foldNms    -  directory name list, 1 x n (list)
    foldPaths  -  directory path list, 1 x n (list)
  """
  foldNms = []
  foldPaths = []

  # each sub file
  for dirname, dirNms, fileNms in os.walk(fold):
    for dirNm in dirNms:
      # file and cmd path
      foldPath = os.path.join(dirname, dirNm)

      # store
      foldNms.append(dirNm)
      foldPaths.append(foldPath)

  return foldNms, foldPaths


def listFile(fold, subx=None):
  """
  Return the list of all files matched with the subfix under a folder.

  Input
    fold       -  root fold
    subx       -  subfix, {None} | 'txt' | ...

  Output
    fileNms    -  directory name list, n x
    filePaths  -  directory path list, n x
  """
  fileNms = []
  filePaths = []

  for foldNm in os.listdir(fold):
    # fold absolute path
    filePath = os.path.join(fold, foldNm)

    # skip non fold
    if os.path.isdir(filePath):
      continue

    # skip filepath
    if subx is not None and not filePath.endswith(subx):
      continue

    # store
    fileNms.append(foldNm)
    filePaths.append(filePath)

  return fileNms, filePaths


def listFileR(fold, subxs=None):
    """
    Return the list of all files matched with the subfix recursively
    under a folder.

    Input
      fold      -  root fold
      subxs     -  subfix array, {None} | [] | 'txt' | ...

    Output
      fileNms    -  directory name list, n x
      filePaths  -  directory path list, n x
    """
    fileNms = []
    filePaths = []

    # each sub file
    for dirname, dirnames, fileNm0s in os.walk(fold, followlinks=True):
      for fileNm in fileNm0s:
        # skip filepath if not matched
        if subxs is not None:
          ok = False
          for subx in subxs:
            if fileNm.endswith(subx):
              ok = True
              break
          if not ok:
            continue

        # file and cmd path
        filePath = os.path.join(dirname, fileNm)

        # store
        fileNms.append(dirname)
        filePaths.append(filePath)

    return fileNms, filePaths


def listFiles(dirNm, isRec, subxs):
  """
  Return the list of all files that have the specified subfix in a folder.

  Input
    dirNm    -  dir fold, string
    isRec    -  flag of being recursive, True | False
    subxs    -  subfix list, 1 x m (list)

  Output
    relDirs  -  directory list
    fileNms  -  file names
  """
  relDirs = []
  fileNms = []

  # recursively
  if isRec:
    for root, dirs, files in os.walk(dirNm):
      for file in files:
        # skip non-matched file
        if not isEndSub(file, subxs):
          continue

        # relative fold
        relDir = root[len(dirNm):]
        if relDir and relDir[0] == '/':
          relDir = relDir[1:]

        # store
        relDirs.append(relDir)
        fileNms.append(file)

  # not recursively
  else:
    for file in os.listdir(dirNm):
      # file absolute path
      filePath = os.path.join(dirNm, file)

      # skip fold on non-matched file
      if os.path.isdir(filePath) or not isEndSub(file, subxs):
        continue

      # store
      relDirs.append('')
      fileNms.append(file)

  return (relDirs, fileNms)


def getch():
  """
  Get one char from input.

  Output
    c  -  character
  """
  import termios
  import sys
  import tty

  fd = sys.stdin.fileno()
  old_settings = termios.tcgetattr(fd)
  try:
    tty.setraw(fd)
    c = sys.stdin.read(1)
  finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
  return c


def loadCsv(csvPath, nLnSkip=0, delimiter=',', quotechar=None):
  """
  Load from csv.

  Input
    csvPath    -  csv path
    nLnSkip    -  #line to skip in the header, {0} | ...
    delimiter  -  delimiter, {','} | ...
    quotechar  -  quotechar, {None} | ...

  Output
    dcts       -  symbol list, n x
    keys       -  key list, nKey x
  """
  dcts = []
  with open(csvPath, 'rb') as csvfile:
    csvHa = csv.reader(csvfile, delimiter=delimiter, quotechar=quotechar)
    for i, row in enumerate(csvHa):
      # skip line
      if i < nLnSkip:
        continue

      # field name
      if i == nLnSkip:
        keys = row
        nKey = len(row)
        continue

      # symbol
      assert(len(row) == nKey)
      dct = {}
      for iKey, key in enumerate(keys):
        dct[key] = row[iKey]
      dcts.append(dct)
  return dcts, keys


def lmdbRIn(lmdbPath):
  """
  Get the lmdb handle of a given sequence.

  Input
    lmdbPath  -  path of the lmdb file

  Output
    ha        -  handles
  """
  import lmdb

  # path file
  env = lmdb.open(lmdbPath)
  txn = env.begin()
  cur = txn.cursor()

  # store
  ha = {'env': env,
        'cur': cur,
        'co': 0,
        'lmdb': lmdbPath}
  return ha


def lmdbR(ha):
  """
  Read one item from the lmdb handle.

  Input
    ha   -  lmdb handle

  Output
    key  -  key
    val  -  value
  """
  # move cursor
  if ha['co'] == 0:
    ha['cur'].first()
  else:
    if not ha['cur'].next():
      return None, None
  ha['co'] += 1

  # get key & value
  key = ha['cur'].key()
  val = ha['cur'].value()

  return key, val


def lmdbROut(ha):
  """
  Close the handler.

  Input
    ha  -  lmdb handle
  """
  ha['env'].close()


def hdfRIn(hdfPath):
  """
  Open an hdf handler.

  Input
    hdfPath  -  hdf path

  Output
    ha       -  handler
  """
  import h5py
  ha = h5py.File(hdfPath, 'r')

  return ha


def hdfR(ha, nm='a'):
  """
  Read from hdf handler.

  Input
    ha  -  hdf handler
    nm  -  name, {'a'}

  Output
    A   -  result
  """
  A0 = ha[nm]

  import numpy as np
  A = np.array(A0)

  return A


def hdfROut(ha):
  """
  Close a HDF handler.

  Input
    ha  -  hdf handler
  """
  ha.close()


def checkFoldMatch(srcDir, dstDir, logFile1, logFile2, isRec, subxs):
  # check subfold match

  # open file
  fio1 = open(logFile1, 'w')
  fio2 = open(logFile2, 'w')

  # file list
  relDirs, fileNms = listFiles(dstDir, isRec, subxs)

  # each file
  for i in range(len(relDirs)):
    relDir = relDirs[i]
    fileNm = fileNms[i]

    # pdb.set_trace()

    # file path
    srcFile = os.path.join(srcDir, relDir, fileNm)
    dstFile = os.path.join(dstDir, relDir, fileNm)

    # matched
    if os.path.isfile(srcFile):
      fio1.write(srcFile + "\n")
      fio1.write(dstFile + "\n")

    # not matched
    else:
      fio2.write(dstFile + "\n");

  # close file
  fio1.close()
  fio2.close()


def isEndSub(filename, subxs):
  """
  Checking whether filename ends with any in the specified subfix list.

  Input
    filename  -  file name
    subxs     -  subfix list, 1 x m (list)

  Output
    res       -  result, True | False
  """
  import re

  for subx in subxs:
    pat = ".*\." + subx + "$"

    if re.match(pat, filename):
      return True

  return False
