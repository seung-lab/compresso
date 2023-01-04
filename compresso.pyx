"""
Python bindings for the Compresso labeled image compression algorithm.

Compatible with format versions 0 and 1.

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
import gzip
import os.path
from libcpp cimport bool as native_bool
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
  void* cpp_decompress(
    unsigned char* buf, size_t num_bytes, void* output, 
    int64_t zstart, int64_t zend
  ) except +
  size_t COMPRESSO_HEADER_SIZE


def compress(data, steps=None, connectivity=4, random_access_z_index=True) -> bytes:
  """
  compress(ndarray[UINT, ndim=3] data, steps=(4,4,1), random_access_z_index=True)

  Compress a 3d numpy array into a compresso byte stream.

  data: 3d ndarray of segmentation labels
  steps: 
    Grid size for classifying the boundary structure.
    Smaller sizes (up to a point) are more likely to compress because 
    they repeat more frequently. (4,4,1) and (8,8,1) are typical.
  connectivity: 4 or 6. 4 means we use 2D connected components and
    6 means we use 3D connected components.
  random_access_z_index: if True, adds an index proportional to the
    size of the z index that enables decoding z slices independently.
    This index is at most 2 * 8 * sz additional bytes. This also changes
    the format version to indicate the index is present.

  Return: compressed bytes b'...'
  """
  explicit_steps = True
  if steps is None:
    steps = (4,4,1)
    explicit_steps = False

  if connectivity not in (4,6):
    raise ValueError(f"{connectivity} connectivity must be 4 or 6.")

  if connectivity == 6:
    random_access_z_index = False

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
    return _compress(data, steps, connectivity, random_access_z_index)
  except RuntimeError as err:
    if "Unable to RLE encode" in str(err) and not explicit_steps:
      return compress(
        data, steps=(8,8,1), 
        connectivity=connectivity, 
        random_access_z_index=random_access_z_index,
      )
    else:
      raise EncodeError(err)

def _compress(
  cnp.ndarray[UINT, ndim=3] data, steps=(4,4,1),
  unsigned int connectivity=4, native_bool random_access_z_index=True
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
    buf = cpp_compress[uint8_t](&arr8[0,0,0], sx, sy, sz, nx, ny, nz, connectivity, random_access_z_index)
  elif data.dtype == np.uint16:
    arr16 = data
    buf = cpp_compress[uint16_t](&arr16[0,0,0], sx, sy, sz, nx, ny, nz, connectivity, random_access_z_index)
  elif data.dtype == np.uint32:
    arr32 = data
    buf = cpp_compress[uint32_t](&arr32[0,0,0], sx, sy, sz, nx, ny, nz, connectivity, random_access_z_index)
  elif data.dtype == np.uint64:
    arr64 = data
    buf = cpp_compress[uint64_t](&arr64[0,0,0], sx, sy, sz, nx, ny, nz, connectivity, random_access_z_index)
  else:
    raise TypeError(f"Type {data.dtype} not supported. Only uints and bool are supported.")

  return bytes(buf)

def check_compatibility(bytes buf):
  format_version = buf[4]
  if format_version > 1:
    raise DecodeError(f"Unable to decode format version {format_version}. Only versions 0 and 1 are supported.")

def label_dtype(dict info):
  """Given a header dict, return the dtype for the labels."""
  dtypes = {
    1: np.uint8,
    2: np.uint16,
    4: np.uint32,
    8: np.uint64,
  }
  return dtypes[info["data_width"]]  

def window_dtype(dict info):
  """Given a header dict, return the dtype for the boundary windows."""
  window_size = info["xstep"] * info["ystep"] * info["zstep"]
  if window_size <= 16:
    return np.uint16
  elif window_size <= 32:
    return np.uint32
  else:
    return np.uint64

def header(bytes buf):
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

def nbytes(bytes buf):
  """Compute the number of bytes the decompressed array will consume."""
  info = header(buf)
  return info["sx"] * info["sy"] * info["sz"] * info["data_width"]

def raw_header(bytes buf):
  """Return the bytes corresponding to the header."""
  return np.frombuffer(buf[:COMPRESSO_HEADER_SIZE], dtype=np.uint8)

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

def raw_labels(buf):
  """Returns the labels array present in the compressed stream."""
  info = header(buf)

  offset = COMPRESSO_HEADER_SIZE
  id_bytes = info["id_size"] * info["data_width"]
  ldtype = label_dtype(info)
  wdtype = window_dtype(info)

  return np.frombuffer(buf[offset:offset+id_bytes], dtype=ldtype)

def raw_z_index(bytes buf):
  """Return the z index if present."""
  info = header(buf)
  format_version = info["format_version"]
  sz = info["sz"]

  if format_version == 0:
    return None

  num_bytes = 2 * 8 * sz
  return np.frombuffer(buf[-num_bytes:], dtype=np.uint64).reshape((2,sz), order="C")

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

def remap(bytes buf, dict mapping, native_bool preserve_missing_labels=False):
  """
  bytes remap(bytes buf, dict mapping, preserve_missing_labels=False)

  Remap the labels of a compresso stream without decompressing.
  """
  ids = np.copy(raw_ids(buf))

  cdef size_t i = 0  
  cdef size_t size = ids.size
  for i in range(size):
    try:
      ids[i] = mapping[ids[i]]
    except KeyError:
      if not preserve_missing_labels:
        raise

  locations = np.copy(raw_locations(buf))
  locations = _remap_locations(locations, mapping, preserve_missing_labels)

  head = raw_header(buf)
  values = raw_values(buf)
  windows = raw_windows(buf)

  return (
    head.tobytes()
    + ids.tobytes()
    + values.tobytes() 
    + locations.tobytes() 
    + windows.tobytes()
  )

def _remap_locations(
  cnp.ndarray[UINT] locations, 
  dict mapping, 
  native_bool preserve_missing_labels
):
  cdef size_t i = 0
  cdef size_t size = locations.size
  while i < size:
    if locations[i] == 6:
      try:
        locations[i+1] = mapping[locations[i+1]]
      except KeyError:
        if not preserve_missing_labels:
          raise
      i += 1
    elif locations[i] > 6:
      try:
        locations[i] = mapping[locations[i] - 7] + 7
      except KeyError:
        if not preserve_missing_labels:
          raise
    i += 1  
  
  return locations

def decompress(bytes data, z=None):
  """
  Decompress a compresso encoded byte stream into a three dimensional 
  numpy array containing image segmentation.

  z: int or (zstart:int, zend:int) to decompress 
    only a single or selected range of z slices.

  Returns: 3d ndarray
  """
  info = header(data)
  sz = info["sz"]

  zstart = 0
  zend = sz
  if isinstance(z, int):
    zstart = z
    zend = z + 1
  elif hasattr(z, "__getitem__"):
    zstart, zend = z[0], z[1]

  if zstart < 0 or zstart > sz:
    raise ValueError(f"zstart must be between 0 and sz - 1 ({sz-1}): {zstart}")
  if zend < 0 or zend > sz:
    raise ValueError(f"zend must be between 1 and sz ({sz}): {zend}")
  if zend < zstart:
    raise ValueError(f"zend ({zend}) must be >= zstart ({zstart})")


  shape = (info["sx"], info["sy"], zend - zstart)

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
    cpp_decompress(buf, len(data), outptr, zstart, zend)
  except RuntimeError as err:
    raise DecodeError(err)

  return labels

def load(filelike):
  """Load an image from a file-like object or file path."""
  if hasattr(filelike, 'read'):
    binary = filelike.read()
  elif (
    isinstance(filelike, str) 
    and os.path.splitext(filelike)[1] == '.gz'
  ):
    with gzip.open(filelike, 'rb') as f:
      binary = f.read()
  else:
    with open(filelike, 'rb') as f:
      binary = f.read()
  return decompress(binary)

def save(labels, filelike):
  """Save labels into the file-like object or file path."""
  binary = compress(labels)
  if hasattr(filelike, 'write'):
    filelike.write(binary)
  elif (
    isinstance(filelike, str) 
    and os.path.splitext(filelike)[1] == '.gz'
  ):
    with gzip.open(filelike, 'wb') as f:
      f.write(binary)
  else:
    with open(filelike, 'wb') as f:
      f.write(binary)

def valid(bytes buf):
  """Does the buffer appear to be a valid compresso stream?"""
  if len(buf) < <Py_ssize_t>COMPRESSO_HEADER_SIZE:
    return False

  head = header(buf)
  if head["magic"] != b"cpso":
    return False

  format_version = head["format_version"]

  if format_version not in (0,1):
    return False

  cdef int window_bits = head["xstep"] * head["ystep"] * head["zstep"]
  cdef int window_bytes = 0
  if window_bits <= 8:
    window_bytes = 1
  elif window_bits <= 16:
    window_bytes = 2
  elif window_bits <= 32:
    window_bytes = 4
  else:
    window_bytes = 8

  zindex_size = 0
  if format_version == 1:
    zindex_size = 2 * head["sz"] * zindex_byte_width(head["sx"], head["sy"])

  min_size = (
    COMPRESSO_HEADER_SIZE 
    + (head["id_size"] * head["data_width"]) 
    + (head["value_size"] * window_bytes)
    + (head["location_size"] * head["data_width"])
    + zindex_size
  )
  if len(buf) < min_size:
    return False

  return True

def zindex_byte_width(sx, sy):
  worst_case = 2 * sx * sy
  if worst_case < 2 ** 8:
    return 1
  elif worst_case < 2 ** 16:
    return 2
  elif worst_case < 2 ** 32:
    return 4
  else:
    return 8

class CompressoArray:
  def __init__(self, binary):
    self.binary = binary

  def __len__(self):
    return len(self.binary)

  @property
  def random_access_enabled(self):
    head = header(self.binary)
    return head["format_version"] == 1

  @property
  def size(self):
    shape = self.shape
    return shape[0] * shape[1] * shape[2]

  @property
  def nbytes(self):
    return nbytes(self.binary)

  @property
  def dtype(self):
    return label_dtype(header(self.binary))

  @property
  def shape(self):
    head = header(self.binary)
    return (head["sx"], head["sy"], head["sz"])

  def labels(self):
    return labels(self.binary)

  def remap(self, buf, mapping, preserve_missing_labels=False):
    return CompressoArray(remap(buf, mapping, preserve_missing_labels))

  def __getitem__(self, slcs):
    slices = reify_slices(slcs, *self.shape)

    if isinstance(slcs, (slice, int)):
      slcs = (slcs,)

    while len(slcs) < 3:
       slcs += (slice(None, None, None),)

    if self.random_access_enabled:
      img = decompress(self.binary, z=(slices[2].start, slices[2].stop))
      zslc = slice(None, None, slices[2].step)
      if hasattr(slcs, "__getitem__") and isinstance(slcs[2], int):
        zslc = 0
      slices = (slcs[0], slcs[1], zslc)
      return img[slices]
    else:
      img = decompress(self.binary)
      return img[slcs]

def reify_slices(slices, sx, sy, sz):
  """
  Convert free attributes of a slice object 
  (e.g. None (arr[:]) or Ellipsis (arr[..., 0]))
  into bound variables in the context of this
  bounding box.

  That is, for a ':' slice, slice.start will be set
  to the value of the respective minpt index of 
  this bounding box while slice.stop will be set 
  to the value of the respective maxpt index.

  Example:
    reify_slices( (np._s[:],) )
    
    >>> [ slice(-1,1,1), slice(-2,2,1), slice(-3,3,1) ]

  Returns: [ slice, ... ]
  """
  ndim = 3
  minpt = (0,0,0)
  maxpt = (sx,sy,sz)

  integer_types = (int, np.integer)
  floating_types = (float, np.floating)

  if isinstance(slices, integer_types) or isinstance(slices, floating_types):
    slices = [ slice(int(slices), int(slices)+1, 1) ]
  elif isinstance(slices, slice):
    slices = [ slices ]
  elif slices is Ellipsis:
    slices = []

  slices = list(slices)

  for index, slc in enumerate(slices):
    if slc is Ellipsis:
      fill = ndim - len(slices) + 1
      slices = slices[:index] +  (fill * [ slice(None, None, None) ]) + slices[index+1:]
      break

  while len(slices) < ndim:
    slices.append( slice(None, None, None) )

  # First three slices are x,y,z, last is channel. 
  # Handle only x,y,z here, channel seperately
  for index, slc in enumerate(slices):
    if isinstance(slc, integer_types) or isinstance(slc, floating_types):
      slices[index] = slice(int(slc), int(slc)+1, 1)
    elif slc == Ellipsis:
      raise ValueError("More than one Ellipsis operator used at once.")
    else:
      start = 0 if slc.start is None else slc.start
      end = maxpt[index] if slc.stop is None else slc.stop 
      step = 1 if slc.step is None else slc.step

      if step < 0:
        raise ValueError(f'Negative step sizes are not supported. Got: {step}')

      if start < 0: # this is support for negative indicies
        start = maxpt[index] + start         
      check_bounds(start, minpt[index], maxpt[index])
      if end < 0: # this is support for negative indicies
        end = maxpt[index] + end
      check_bounds(end, minpt[index], maxpt[index])

      slices[index] = slice(start, end, step)

  return slices

def clamp(val, low, high):
  return min(max(val, low), high)

def check_bounds(val, low, high):
  if val > high or val < low:
    raise ValueError(f'Value {val} cannot be outside of inclusive range {low} to {high}')
  return val

