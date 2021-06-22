"""
Python bindings for the Compresso labeled image compression algorithm.

Compatible with format version 0.

B. Matejek, D. Haehn, F. Lekschas, M. Mitzenmacher, and H. Pfister.
"Compresso: Efficient Compression of Segmentation Data for Connectomics".
Springer: Intl. Conf. on Medical Image Computing and Computer-Assisted Intervention.
2017.

Modifications by William Silversmith.

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

class EncodeError(Exception):
  """Unable to encode the stream."""
  pass

class DecodeError(Exception):
  """Unable to decode the stream."""
  pass

cdef extern from "compresso.hpp" namespace "pycompresso":
  vector[unsigned char] cpp_zero_data_stream(
    size_t sx, size_t sy, size_t sz, 
    size_t xstep, size_t ystep, size_t zstep,
    size_t data_width, size_t connectivity
  )
  vector[unsigned char] cpp_compress[T](
    T *data, 
    size_t sx, size_t sy, size_t sz, 
    size_t xstep, size_t ystep, size_t zstep,
    size_t connectivity
  ) except +
  void* cpp_decompress(unsigned char* buf, size_t num_bytes, void* output) except +
  size_t COMPRESSO_HEADER_SIZE


def compress(data, steps=None, connectivity=4) -> bytes:
  """
  compress(ndarray[UINT, ndim=3] data, steps=(4,4,1))

  Compress a 3d numpy array into a compresso byte stream.

  data: 3d ndarray of segmentation labels
  steps: 
    Grid size for classifying the boundary structure.
    Smaller sizes (up to a point) are more likely to compress because 
    they repeat more frequently. (4,4,1) and (8,8,1) are typical.
  connectivity: 4 or 6. 4 means we use 2D connected components and
    6 means we use 3D connected components.

  Return: compressed bytes b'...'
  """
  explicit_steps = True
  if steps is None:
    steps = (4,4,1)
    explicit_steps = False

  if connectivity not in (4,6):
    raise ValueError(f"{connectivity} connectivity must be 4 or 6.")

  while data.ndim > 3:
    if data.shape[-1] == 1:
      data = data[..., 0]
    else:
      break

  if data.ndim > 3:
    raise TypeError(f"Image must be at most three dimensional. Got {data.ndim} dimensions.")
  
  while data.ndim < 3:
    data = data[..., np.newaxis]

  sx, sy, sz = data.shape
  nx, ny, nz = steps
  data_width = np.dtype(data.dtype).itemsize

  if data.size == 0:
    return bytes(cpp_zero_data_stream(sx, sy, sz, nx, ny, nz, data_width, connectivity))

  data = np.asfortranarray(data)

  try:
    return _compress(data, steps, connectivity)
  except RuntimeError as err:
    if "Unable to RLE encode" in str(err) and not explicit_steps:
      return compress(data, steps=(8,8,1), connectivity=connectivity)
    else:
      raise EncodeError(err)

def _compress(
  cnp.ndarray[UINT, ndim=3] data, steps=(4,4,1),
  unsigned int connectivity=4
) -> bytes:
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
    buf = cpp_compress[uint8_t](&arr8[0,0,0], sx, sy, sz, nx, ny, nz, connectivity)
  elif data.dtype == np.uint16:
    arr16 = data
    buf = cpp_compress[uint16_t](&arr16[0,0,0], sx, sy, sz, nx, ny, nz, connectivity)
  elif data.dtype == np.uint32:
    arr32 = data
    buf = cpp_compress[uint32_t](&arr32[0,0,0], sx, sy, sz, nx, ny, nz, connectivity)
  elif data.dtype == np.uint64:
    arr64 = data
    buf = cpp_compress[uint64_t](&arr64[0,0,0], sx, sy, sz, nx, ny, nz, connectivity)
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
  window_size = info["xstep"] * info["ystep"] * info["zstep"]
  if window_size <= 16:
    return np.uint16
  elif window_size <= 32:
    return np.uint32
  else:
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
    "connectivity": buf[35],
  }

def nbytes(buf : bytes):
  """Compute the number of bytes the decompressed array will consume."""
  info = header(buf)
  return info["sx"] * info["sy"] * info["sz"] * info["data_width"]

def raw_ids(bytes buf):
  """Return the ids buffer from the compressed stream."""
  info = header(buf)

  offset = COMPRESSO_HEADER_SIZE
  id_bytes = info["id_size"] * info["data_width"]
  ldtype = label_dtype(info)
  return np.frombuffer(buf[offset:offset+id_bytes], dtype=ldtype)

def raw_values(bytes buf):
  """Return the window values buffer from the compressed stream."""
  info = header(buf)
  
  id_bytes = info["id_size"] * info["data_width"]
  ldtype = label_dtype(info)
  wdtype = window_dtype(info)

  offset = COMPRESSO_HEADER_SIZE + id_bytes
  value_bytes = info["value_size"] * np.dtype(wdtype).itemsize
  
  return np.frombuffer(buf[offset:offset+value_bytes], dtype=wdtype)

def raw_locations(bytes buf):
  """Return the indeterminate locations buffer from the compressed stream."""
  info = header(buf)

  offset = COMPRESSO_HEADER_SIZE
  id_bytes = info["id_size"] * info["data_width"]
  ldtype = label_dtype(info)
  wdtype = window_dtype(info)

  value_bytes = info["value_size"] * np.dtype(wdtype).itemsize
  
  offset += id_bytes + value_bytes
  location_bytes = info["location_size"] * info["data_width"]
  return np.frombuffer(buf[offset:offset+location_bytes], dtype=ldtype)

def raw_windows(bytes buf):
  """Return the window boundary data buffer from the compressed stream."""
  info = header(buf)

  ldtype = label_dtype(info)
  wdtype = window_dtype(info)

  id_bytes = info["id_size"] * info["data_width"]
  value_bytes = info["value_size"] * np.dtype(wdtype).itemsize
  location_bytes = info["location_size"] * info["data_width"]

  offset = COMPRESSO_HEADER_SIZE + id_bytes + value_bytes + location_bytes

  return np.frombuffer(buf[offset:], dtype=wdtype)

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
  
  decoded_locations = np.zeros((locations.size,), dtype=ldtype)
  decoded = _extract_labels_from_locations(locations, decoded_locations)

  labels = np.concatenate((ids, decoded_locations[:decoded]))
  return np.unique(labels)

def _extract_labels_from_locations(
  cnp.ndarray[UINT, ndim=1] locations, 
  cnp.ndarray[UINT, ndim=1] decoded_locations,
):
  """Helper function for labels."""
  cdef size_t i = 0
  cdef size_t j = 0
  cdef size_t sz = locations.size
  while i < sz:
    if locations[i] == 6:
      decoded_locations[j] = locations[i+1]
      i += 1
      j += 1
    elif locations[i] > 6:
      decoded_locations[j] = locations[i] - 7
      j += 1
    i += 1

  return j # size of decoded_locations

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

  if labels.size == 0:
    return labels

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

  if outptr == NULL:
    raise DecodeError("Unable to decode stream.")

  cdef unsigned char* buf = data
  try:
    cpp_decompress(buf, len(data), outptr)
  except RuntimeError as err:
    raise DecodeError(err)

  return labels










