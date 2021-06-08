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
  void* cpp_decompress(unsigned char* buf, size_t num_bytes, void* output)
  size_t COMPRESSO_HEADER_SIZE

def compress(cnp.ndarray[UINT, ndim=3] data, steps=(4,4,1)) -> bytes:
  """
  compress(ndarray[UINT, ndim=3] data, steps=(4,4,1))

  Compress a 3d numpy array into a compresso byte stream.

  data: 3d ndarray of segmentation labels
  steps: 
    Grid size for classifying the boundary structure.
    Smaller sizes (up to a point) are more likely to compress because 
    they repeat more frequently. (4,4,1) and (8,8,1) are typical.

  Return: compressed bytes b'...'
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

def label_dtype(info : dict):
  """Given a header dict, return the dtype for the labels."""
  dtypes = {
    1: np.uint8,
    2: np.uint16,
    4: np.uint32,
    8: np.uint64,
  }
  return dtypes[info["data_width"]]  

def window_dtype(info : dict):
  """Given a header dict, return the dtype for the boundary windows."""
  if info["xstep"] * info["ystep"] * info["zstep"] <= 16:
    return np.uint16
  return np.uint64

def header(buf : bytes) -> dict:
  """
  Decodes the header into a python dict.
  """
  check_compatibility(buf)
  toint = lambda n: int.from_bytes(n, byteorder="little", signed=False)

  return {
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

def nbytes(buf : bytes):
  """Compute the number of bytes the decompressed array will consume."""
  info = header(buf)
  return info["sx"] * info["sy"] * info["sz"] * info["data_width"]

def labels(bytes buf):
  """
  Returns a sorted list of the unique labels
  in this stream without decompressing to
  a full 3D array. Faster and lower memory.

  This data can be retrieved from the ids
  field and the locations field.
  """
  info = header(buf)

  offset = COMPRESSO_HEADER_SIZE
  id_bytes = info["id_size"] * info["data_width"]
  ldtype = label_dtype(info)
  wdtype = window_dtype(info)

  ids = np.frombuffer(buf[offset:offset+id_bytes], dtype=ldtype)

  value_bytes = info["value_size"] * np.dtype(wdtype).itemsize
  
  offset += id_bytes + value_bytes
  location_bytes = info["location_size"] * info["data_width"]
  locations = np.frombuffer(buf[offset:offset+location_bytes], dtype=ldtype)
  locations = locations[locations >= 7] - 7

  labels = np.concatenate((ids, locations))
  return np.unique(labels[labels > 0])

def decompress(bytes data):
  """
  Decompress a compresso encoded byte stream into a three dimensional 
  numpy array containing image segmentation.

  Returns: 3d ndarray
  """
  info = header(data)
  shape = (info["sx"], info["sy"], info["sz"])

  dtype = label_dtype(info)
  labels = np.zeros(shape, dtype=dtype, order="F")

  cdef cnp.ndarray[uint8_t, ndim=3] labels8
  cdef cnp.ndarray[uint16_t, ndim=3] labels16
  cdef cnp.ndarray[uint32_t, ndim=3] labels32
  cdef cnp.ndarray[uint64_t, ndim=3] labels64

  cdef void* outptr
  if dtype == np.uint8:
    labels8 = labels
    outptr = <void*>&labels8[0,0,0]
  elif dtype == np.uint16:
    labels16 = labels
    outptr = <void*>&labels16[0,0,0]
  elif dtype == np.uint32:
    labels32 = labels
    outptr = <void*>&labels32[0,0,0]
  else:
    labels64 = labels
    outptr = <void*>&labels64[0,0,0]

  cdef unsigned char* buf = data
  cpp_decompress(buf, len(data), outptr)

  return labels










