import pytest

import numpy as np
import compresso

DTYPES = [
  np.uint8, np.uint16, np.uint32, np.uint64,
]

@pytest.mark.parametrize('dtype', DTYPES)
def test_empty(dtype):
  labels = np.zeros((0,0,0), dtype=dtype, order="F")
  compressed = compresso.compress(labels)

  reconstituted = compresso.decompress(compressed)
  assert np.all(labels == reconstituted)

@pytest.mark.parametrize('dtype', DTYPES)
def test_uniform_field(dtype):
  labels = np.zeros((100,100,100), dtype=dtype, order="F") + 1
  compressed = compresso.compress(labels)
  reconstituted = compresso.decompress(compressed)
  assert len(compressed) < labels.nbytes
  assert np.all(labels == reconstituted)

  labels = np.zeros((100,100,100), dtype=dtype, order="F") + np.iinfo(dtype).max
  compressed2 = compresso.compress(labels)
  reconstituted = compresso.decompress(compressed2)
  assert len(compressed2) < labels.nbytes
  assert np.all(labels == reconstituted)

@pytest.mark.parametrize('dtype', DTYPES)
def test_arange_field(dtype):
  labels = np.arange(0,1024).reshape((16,16,4)).astype(dtype)
  compressed = compresso.compress(labels)
  reconstituted = compresso.decompress(compressed)
  assert np.all(labels == reconstituted)

  labels = np.arange(1,1025).reshape((16,16,4)).astype(dtype)
  compressed = compresso.compress(labels)
  reconstituted = compresso.decompress(compressed)
  assert np.all(labels == reconstituted)

@pytest.mark.parametrize('dtype', DTYPES)
def test_2d_arange_field(dtype):
  labels = np.arange(0,16*16).reshape((16,16,1)).astype(dtype)
  compressed = compresso.compress(labels)
  reconstituted = compresso.decompress(compressed)
  assert np.all(labels == reconstituted)

@pytest.mark.parametrize('dtype', DTYPES)
def test_2_field(dtype):
  labels = np.arange(0,1024).reshape((16,16,4)).astype(dtype)
  compressed = compresso.compress(labels)
  reconstituted = compresso.decompress(compressed)
  assert np.all(labels == reconstituted)
  
  labels[2,2,1] = np.iinfo(dtype).max
  compressed = compresso.compress(labels)
  reconstituted = compresso.decompress(compressed)
  assert np.all(labels == reconstituted)

@pytest.mark.parametrize('order', ("C", "F"))
@pytest.mark.parametrize('dtype', DTYPES)
def test_random_field(dtype, order):
  labels = np.random.randint(0, 25, size=(100, 100, 25)).astype(dtype)
  
  if order == "C":
    labels = np.ascontiguousarray(labels)
  else:
    labels = np.asfortranarray(labels)

  compressed = compresso.compress(labels)

  reconstituted = compresso.decompress(compressed)
  assert np.all(labels == reconstituted)
