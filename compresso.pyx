"""
Python bindings for the Compresso labeled image compression algorithm.

B. Matejek, D. Haehn, F. Lekschas, M. Mitzenmacher, and H. Pfister.
"Compresso: Efficient Compression of Segmentation Data for Connectomics".
Springer: Intl. Conf. on Medical Image Computing and Computer-Assisted Intervention.
2017.

https://vcg.seas.harvard.edu/publications/compresso-efficient-compression-of-segmentation-data-for-connectomics
https://github.com/vcg/compresso

PyPI Distribution: 
https://github.com/seung-lab/compresso

License: MIT
"""

cimport cython
cimport numpy as cnp
import numpy as np
import ctypes
from libcpp.vector cimport vector
from libc.stdint cimport (
  int8_t, int16_t, int32_t, int64_t,
  uint8_t, uint16_t, uint32_t, uint64_t,
)

ctypedef fused UINT:
  uint8_t
  uint16_t
  uint32_t
  uint64_t

class DecodeError(Exception):
  """Unable to decode the stream."""
  pass

cdef extern from "compresso.hxx" namespace "pycompresso":
  vector[unsigned char] cpp_compress[T](
    T *data, 
    size_t sx, size_t sy, size_t sz, 
    size_t xstep, size_t ystep, size_t zstep
  )
  void* cpp_decompress(unsigned char* buf, void* output)

def compress(cnp.ndarray[UINT, ndim=3] data, steps=(4,4,1)) -> bytes:
  """
  compress(ndarray[UINT, ndim=3] data, steps=(4,4,1))

  Compress a 3d numpy array into a compresso byte stream.

  data: 3d ndarray of segmentation labels
  steps: 
    Grid size for classifying the boundary structure.
    Smaller sizes (up to a point) are more likely to compress because 
    they repeat more frequently. (4,4,1) and (8,8,1) are typical.

  Return: ndarray
  """
  data = np.asfortranarray(data)
  sx = data.shape[0]
  sy = data.shape[1]
  sz = data.shape[2]

  nx, ny, nz = steps

  cdef uint8_t[:,:,:] arr8
  cdef uint16_t[:,:,:] arr16
  cdef uint32_t[:,:,:] arr32
  cdef uint64_t[:,:,:] arr64

  cdef vector[unsigned char] buf

  if data.dtype in (np.uint8, bool):
    arr8 = data.view(np.uint8)
    buf = cpp_compress[uint8_t](&arr8[0,0,0], sx, sy, sz, nx, ny, nz)
  elif data.dtype == np.uint16:
    arr16 = data
    buf = cpp_compress[uint16_t](&arr16[0,0,0], sx, sy, sz, nx, ny, nz)
  elif data.dtype == np.uint32:
    arr32 = data
    buf = cpp_compress[uint32_t](&arr32[0,0,0], sx, sy, sz, nx, ny, nz)
  elif data.dtype == np.uint64:
    arr64 = data
    buf = cpp_compress[uint64_t](&arr64[0,0,0], sx, sy, sz, nx, ny, nz)
  else:
    raise TypeError(f"Type {data.dtype} not supported. Only uints and bool are supported.")

  return bytes(buf)

def check_compatibility(buf : bytes):
  format_version = buf[4]
  if format_version != 0:
    raise DecodeError(f"Unable to decode format version {format_version}. Only version 0 is supported.")

def read_header(buf : bytes) -> dict:
  """
  Decodes the header into a python dict.
  """
  check_compatibility(buf)
  toint = lambda n: int.from_bytes(n, byteorder="little", signed=False)

  data = {
    "magic": buf[:4],
    "format_version": buf[4],
    "data_width": buf[5],
    "sx": toint(buf[6:8]),
    "sy": toint(buf[8:10]),
    "sz": toint(buf[10:12]),
    "xstep": buf[12],
    "ystep": buf[13],
    "zstep": buf[14],
    "id_size": toint(buf[15:23]),
    "value_size": toint(buf[23:27]),
    "location_size": toint(buf[27:35]),
  }
  data["decompressed_bytes"] = data["sx"] * data["sy"] * data["sz"] * data["data_width"]

def decompress(bytes data):
  """
  Decompress a compresso encoded byte stream into a three dimensional 
  numpy array containing image segmentation.

  Returns: compressed bytes b'...'
  """
  check_compatibility(data)
  header = read_header(data)
  shape = (header["sx"], header["sy"], header["sz"])

  dtypes = {
    1: np.uint8,
    2: np.uint16,
    4: np.uint32,
    8: np.uint64,
  }
  dtype = dtypes[header["data_width"]]
  cdef cnp.ndarray[uint64_t, ndim=3] labels = np.zeros(shape, dtype=dtype, order="F")

  cdef unsigned char* buf = data
  cpp_decompress(buf, <void*>&labels[0,0,0])

  return labels










