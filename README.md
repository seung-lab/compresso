# Compresso: Efficient Compression of Segmentation Data For Connectomics (PyPI edition)

[![PyPI version](https://badge.fury.io/py/compresso.svg)](https://badge.fury.io/py/compresso)
[![Paper](https://img.shields.io/badge/paper-accepted-red.svg?colorB=f52ef0)](https://vcg.seas.harvard.edu/publications/compresso-efficient-compression-of-segmentation-data-for-connectomics)
[![MICCAI](https://img.shields.io/badge/presentation-MICCAI%202017-red.svg?colorB=135f89)](http://www.miccai2017.org/schedule)


![Segmentations](/banner.png?raw=true)

```python
import compresso 
import numpy as np 

labels = np.array(...)
compressed_labels = compresso.compress(labels) # 3d numpy array -> compressed bytes
reconstituted_labels = compresso.decompress(compressed_labels) # compressed bytes -> 3d numpy array

# Returns header info as dict
# Has array dimensions and data width information.
header = compresso.header(compressed_labels) 

# Extract the unique labels from a stream without 
# decompressing to a full 3D array. Fast and low memory.
uniq_labels = compresso.labels(compressed_labels)
```

```bash
# CLI compression of numpy data
# Compresso is designed to use a second stage compressor
# so use gzip, lzma, or others on the output file.
$ compresso data.npy # -> data.npy.cpso
$ compresso -d data.npy.cpso # -> data.npy
$ compresso --help
```

*NOTE: This is an extensive modification of the work by Matejek et al. which can be found here: https://github.com/VCG/compresso. It is not compatible with RhoANA streams.*

> Recent advances in segmentation methods for connectomics and biomedical imaging produce very large datasets with labels that assign object classes to image pixels. The resulting label volumes are bigger than the raw image data and need compression for efficient storage and transfer. General-purpose compression methods are less effective because the label data consists of large low-frequency regions with structured boundaries unlike natural image data. We present Compresso, a new compression scheme for label data that outperforms existing approaches by using a sliding window to exploit redundancy across border regions in 2D and 3D. We compare our method to existing compression schemes and provide a detailed evaluation on eleven biomedical and image segmentation datasets. Our method provides a factor of 600-2200x compression for label volumes, with running times suitable for practice.

**Paper**: Matejek _et al._, "Compresso: Efficient Compression of Segmentation Data For Connectomics", Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2017, 10-14. \[[CITE](https://scholar.google.com/scholar?q=Compresso%3A+Efficient+Compression+of+Segmentation+Data+For+Connectomics) | [PDF](https://vcg.seas.harvard.edu/publications/compresso-efficient-compression-of-segmentation-data-for-connectomics/paper)\]

In more concrete but simple terms, compresso represents the boundary between segments as a boolean bit packed field. Long runs of zeros are run length encoded. The 4-connected components within that field are mapped to a corresponding label. Boundary voxels are decoded with reference to their neighbors, or if that fails, by storing their label. A second stage of compression is then applied, such as gzip or lzma. There's a few more details but that's a reasonable overview.

## Setup

Requires Python 3.6+

```bash
pip install compresso
```

## Versions

| Major Version | Format Version | Description                                                    |
|---------------|----------------|----------------------------------------------------------------|
| 1             | -              | Initial Release. Not usable due to bugs. No format versioning. |
| 2             | 0              | First major release.                                           |

## Compresso Stream Format

| Section   | Bytes                                     | Description                                                                                                     |
|-----------|-------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| Header    | 36                                        | Metadata incl. length of fields.                                                                                |
| ids       | header.data_width * header.id_size        | Map of CCL regions to labels.                                                                                   |
| values    | window_size * header.value_size           | Values of renumbered windows. Bitfields describing boundaries.                                                  |
| locations | header.data_width * header.locations_size | Sequence of 7 control codes and labels + 7 that describe how to decode indeterminate locations in the boundary. |
| windows   | The rest of the stream.                   | Sequence of numbers to be remapped from values. Describes the boundary structure of labels.                     |

`window_size` is the smallest data type that will contain `xstep * ystep * zstep`. For example, `steps=(4,4,1)` uses uint16 while `steps=(8,8,1)` uses uint64.

## Codec Changes

The original codec has been updated and is no longer compatible with the original. Below are the important changes we made that differ from the code published alongside the paper. 

Implementation wise, we also fixed up several bugs, added guards against data corruption, did some performance tuning, and made sure that the entire codec is implemented in C++ and called by Python. Thus, the codec is usable in both C++ and Python as well as any languages, such as Web Assembly, that C++ can be transpiled to. 

Thank you to the original authors for publishing your code and algorithm from which this repo is derived.

### Updated Header

The previous header was 72 bytes. We updated the header to be only 35 bytes. It now includes the magic number `cpso`, a version number, and the data width of the labels. 
This additional information makes detecting valid compresso streams easier, allows for updating the format in the future, and allows us to assume smaller byte widths than 64-bit.  

| Attribute         | Value             | Type    | Description                                     |
|-------------------|-------------------|---------|-------------------------------------------------|
| magic             | cpso              | char[4] | File magic number.                              |
| format_version    | 0                 | u8      | Version of the compresso stream.                |
| data_width        | 1,2,4,or 8        | u8      | Size of the labels in bytes.                    |
| sx, sy, sz        | >= 0              | u16 x 3 | Size of array dimensions.                       |
| xstep,ystep,zstep | 0 < product <= 64 | u8 x 3  | Size of structure grid.                         |
| id_size           | >= 0              | u64     | Size of array mapping of CCL regions to labels. |
| value_size        | >= 0              | u32     | Size of array mapping windows to renumbering.   |
| location_size     | >= 0              | u64     | Size of indeterminate locations array.          |
| connectivity      | 4 or 6            | u8      | Connectivity for connected components.          |

### Char Byte Stream 

The previous implementation treated the byte stream as uniform u64 little endian. We now emit the encoded stream as `unsigned char` and write each appropriate data type in little endian.

### Variable Data Widths

The labels may assume any unsigned integer data width, which reduces the size of the ids and locations stream when appropriate. The encoded boundaries are reduced to the smallest size that fits. A 4x4x1 window is represented with u16, an 8x8x1 with u64. Less commonly used, but a 4x4x2 would be represented with u32, and a 4x2x1 would get a u8.

*Note that at this time only 4x4x1 and 8x8x1 are supported in this implementation, but the protocol will make those assumptions.*

### Supports Full Integer Range in Indeterminate Locations

The previous codec reserved 6 integers for instructions in the locations stream, but this meant that six segmentation labels were not representable. We added a seventh reserved instruction that indicates the next byte in the stream is the label and then we can use the full range of the integer to represent that number.

This potentially expands the size of the compressed stream. However, we only use this instruction for non-representable numbers, so for most data it should cause zero increase and minimal increase so long as the non-representable numbers in indeterminate locations are rare. The upside is compresso now handles all possible inputs.

### Supports 4 and 6 Connected Components

6-connected CCL seems like it would be a win because it would reduce the number of duplicated IDs that need to be stored. However, in an experiment we found that it did significantly decrease IDs, but at the expense of adding many more boundary voxels (since you need to consider the Z direction now) and increasing the number of indeterminate locations far more. It ended up being slower and larger on some connectomics segmentation we experimented with. 

However, we suspect that there are some images where 6 would do better. An obvious example is a solid
color image that has no boundaries. The images where 6 shines will probably have sparser and straighter boundaries so that fewer additional boundary voxels are introduced.




### Results From the Paper

**Compression Performance**

![Compression Performance of Connectomics Datasets](compression-performance.png?raw=true)

Compression ratios of general-purpose compression methods combined with Compresso and Neuroglancer. Compresso paired with LZMA yields the best compression ratios for all connectomics datasets (left) and in average (four out of five) for the others (right).
