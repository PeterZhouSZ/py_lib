#!/usr/bin/env python
"""
Image-related utility functions.

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2015-03
  modify  -  Feng Zhou (zhfe99@gmail.com), 2015-12
"""
import matplotlib.pyplot as plt


def shImg(img, isFilt=False, ax=None, cmap=None):
  """
  Show image.

  Input
    img     -  image, h x w x 3
    ax      -  axes to show, {None}
    isFilt  -  filt or not, {False}
  """
  if ax is not None:
    fig = plt.gcf()
    fig.sca(ax)

  if isFilt:
    img -= img.min()
    img /= img.max()
    img = img.transpose(1, 2, 0)

  if len(img.shape) == 2 or img.shape[2] > 1:
    plt.imshow(img, cmap=cmap)
  else:
    plt.imshow(img[:, :, 0], cmap=cmap)

  _ = plt.axis('off')


def shImgs(imgs, axs, labs=None):
  """
  Show multiple images.

  Input
    imgs  -  images, n x 1 (numpy array)
    axs   -  axes, n x 1 (numpy array)
  """

  for i, (img, ax) in enumerate(zip(imgs.flatten(), axs.flatten())):
      shImg(img, ax=ax)

      if labs is not None:
          plt.title(labs[i])


def shSvFold(fold, prex, type='pdf'):
  """
  Save image to specified folder with specified prefix.

  Input
    fold  -  image fold
    prex  -  image prefix
    type  -  type, {'pdf'} | 'png' | 'jpg'
  """
  from py_lib.fio import savePath
  imgPath = savePath(fold, prex)
  shSvPath(imgPath, type=type)


def shSvPath(imgPath, type='pdf', dpi=None):
  """
  Save image.

  Input
    imgPath  -  image path
    type     -  type, {'pdf'} | 'png' | 'jpg'
    dpi      -  dpi, {None} | ...
  """
  from py_lib.str import strDelSub
  imgNm = strDelSub(imgPath)

  if dpi is None:
    plt.savefig('{}.{}'.format(imgNm, type), format=type)
  else:
    plt.savefig('{}.{}'.format(imgNm, type), format=type, dpi=dpi)


def shBox(Box, cl):
  """
  Show a bounding box on 2D image.

  Input
    Box  -  bounding box, 2 x 2
              Box[0, 0]: top y
              Box[0, 1]: left x
              Box[1, 0]: bottom y
              Box[1, 1]: right x
    cl   -  color, {'r'} | ...

  Output
    ha   -  box handle
  """
  # bounding box
  yHd = Box[0, 0]
  xHd = Box[0, 1]
  yEd = Box[1, 0]
  xEd = Box[1, 1]

  ha = plt.plot([xHd, xEd, xEd, xHd, xHd], [yHd, yHd, yEd, yEd, yHd], '-', color=cl)

  return ha
