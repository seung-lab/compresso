import pytest
import gzip
from io import BytesIO

import numpy as np
import compresso

DTYPES = [
  np.uint8, np.uint16, np.uint32, np.uint64,
]
STEPS = [
  (4,4,1), (5,5,1), (8,8,1),
  (4,4,2), (5,5,2)
]
CONNECTIVITY = (4,6)

@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('steps', STEPS)
@pytest.mark.parametrize('connectivity', CONNECTIVITY)
def test_empty(dtype, steps, connectivity):
  labels = np.zeros((0,0,0), dtype=dtype, order="F")
  compressed = compresso.compress(labels, steps=steps, connectivity=connectivity)
  reconstituted = compresso.decompress(compressed)
  assert np.all(labels == reconstituted)
  assert np.all(np.unique(labels) == compresso.labels(compressed))

@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('steps', STEPS)
@pytest.mark.parametrize('connectivity', CONNECTIVITY)
def test_black(dtype, steps, connectivity):
  labels = np.zeros((100,100,100), dtype=dtype, order="F")
  compressed = compresso.compress(labels, steps=steps, connectivity=connectivity)
  reconstituted = compresso.decompress(compressed)
  assert np.all(labels == reconstituted)
  assert np.all(np.unique(labels) == compresso.labels(compressed))

@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('steps', STEPS)
@pytest.mark.parametrize('connectivity', CONNECTIVITY)
def test_uniform_field(dtype, steps, connectivity):
  labels = np.zeros((100,100,100), dtype=dtype, order="F") + 1
  compressed = compresso.compress(labels, steps=steps, connectivity=connectivity)
  reconstituted = compresso.decompress(compressed)
  assert len(compressed) < labels.nbytes
  assert np.all(labels == reconstituted)
  assert np.all(np.unique(labels) == compresso.labels(compressed))

  labels = np.zeros((100,100,100), dtype=dtype, order="F") + np.iinfo(dtype).max
  compressed2 = compresso.compress(labels, steps=steps, connectivity=connectivity)
  reconstituted = compresso.decompress(compressed2)
  assert len(compressed2) < labels.nbytes
  assert np.all(labels == reconstituted)
  assert np.all(np.unique(labels) == compresso.labels(compressed2))

@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('steps', STEPS)
@pytest.mark.parametrize('connectivity', CONNECTIVITY)
def test_arange_field(dtype, steps, connectivity):
  labels = np.arange(0,1024).reshape((16,16,4)).astype(dtype)
  compressed = compresso.compress(labels, steps=steps, connectivity=connectivity)
  reconstituted = compresso.decompress(compressed)
  assert np.all(labels == reconstituted)
  assert np.all(np.unique(labels) == compresso.labels(compressed))

  labels = np.arange(1,1025).reshape((16,16,4)).astype(dtype)
  compressed = compresso.compress(labels, connectivity=connectivity)
  reconstituted = compresso.decompress(compressed)
  assert np.all(labels == reconstituted)
  assert np.all(np.unique(labels) == compresso.labels(compressed))

@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('steps', STEPS)
@pytest.mark.parametrize('connectivity', CONNECTIVITY)
def test_2d_arange_field(dtype, steps, connectivity):
  labels = np.arange(0,16*16).reshape((16,16,1)).astype(dtype)
  compressed = compresso.compress(labels, steps=steps, connectivity=connectivity)
  reconstituted = compresso.decompress(compressed)
  assert np.all(labels == reconstituted)
  assert np.all(np.unique(labels) == compresso.labels(compressed))

@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('steps', STEPS)
@pytest.mark.parametrize('connectivity', CONNECTIVITY)
def test_2_field(dtype, steps, connectivity):
  labels = np.arange(0,1024).reshape((16,16,4)).astype(dtype)
  compressed = compresso.compress(labels, steps=steps, connectivity=connectivity)
  reconstituted = compresso.decompress(compressed)
  assert np.all(labels == reconstituted)
  assert np.all(np.unique(labels) == compresso.labels(compressed))
  
  labels[2,2,1] = np.iinfo(dtype).max
  compressed = compresso.compress(labels, steps=steps, connectivity=connectivity)
  reconstituted = compresso.decompress(compressed)
  assert np.all(labels == reconstituted)
  assert np.all(np.unique(labels) == compresso.labels(compressed))

@pytest.mark.parametrize('order', ("C", "F"))
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('steps', STEPS)
@pytest.mark.parametrize('connectivity', CONNECTIVITY)
def test_random_field(dtype, order, steps, connectivity):
  labels = np.random.randint(0, 25, size=(100, 100, 25)).astype(dtype)
  
  if order == "C":
    labels = np.ascontiguousarray(labels)
  else:
    labels = np.asfortranarray(labels)

  compressed = compresso.compress(labels, steps=steps, connectivity=connectivity)

  reconstituted = compresso.decompress(compressed)
  assert np.all(labels == reconstituted)

  assert np.all(np.unique(labels) == compresso.labels(compressed))

# Watershed volumes can blow out the RLE encoding
# due to too many windows.
def test_watershed():
  with gzip.open("./ws.npy.cpso.gz", "rb") as f:
    binary = f.read()

  labels = compresso.decompress(binary)
  del binary

  binary = compresso.compress(labels)
  head = compresso.header(binary)
  assert head["xstep"] == 8
  assert head["ystep"] == 8
  assert head["zstep"] == 1

  try:
    binary = compresso.compress(labels, steps=(4,4,1))
    assert False
  except compresso.EncodeError:
    pass

# This volume blew out a previous logic error in
# the RLE encoder where zero overflow was not 
# handled correctly.
def test_rle_overflow():
  with gzip.open("./rle_defect.npy.gz", "rb") as f:
    binary = BytesIO(f.read())

  labels = np.load(binary)

  binary = compresso.compress(labels, steps=(4,4,1))
  labels = compresso.decompress(binary)

  
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('steps', STEPS)
@pytest.mark.parametrize('connectivity', CONNECTIVITY)
def test_remap(dtype, steps, connectivity):
  labels = np.random.randint(0, 15, size=(64,63,61)).astype(dtype)

  remap = { i: i+20 for i in range(15) }

  binary = compresso.compress(labels)
  assert np.all(compresso.labels(binary) == list(range(15)))

  binary2 = compresso.remap(binary, remap)
  assert np.all(compresso.labels(binary2) == list(range(20, 35)))






