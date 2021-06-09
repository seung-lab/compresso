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

@pytest.mark.parametrize('order', ("C", "F"))
@pytest.mark.parametrize('dtype', DTYPES)
def test_compress_decompress(dtype, order):
  labels = np.random.randint(0, 25, size=(100, 100, 25)).astype(dtype)
  
  if order == "C":
    labels = np.ascontiguousarray(labels)
  else:
    labels = np.asfortranarray(labels)

  compressed = compresso.compress(labels)

  reconstituted = compresso.decompress(compressed)
  assert np.all(labels == reconstituted)
