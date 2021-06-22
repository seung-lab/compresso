/* This is an implementation of the Compresso
 * segmentation compression codec. This 
 * is a heavily modified form of the code 
 * originally written by Brian Matejek. 
 *
 * The stream written by this library is not
 * compatible with the original version. It
 * includes some byte width optimizations 
 * and additional header fields in the output
 * and various functions have been somewhat
 * tuned for speed. It also has a modified 
 * indeterminate locations algorithm to accomodate
 * any possible input.
 *
 * You can find the Compresso paper here:
 * https://vcg.seas.harvard.edu/publications/compresso-efficient-compression-of-segmentation-data-for-connectomics
 *
 * You can find the original code here:
 * https://github.com/VCG/compresso/blob/8378346c9a189a48bf9054c5296ceeb7139634c5/experiments/compression/compresso/cpp-compresso.cpp
 *
 * William Silversmith 
 * Princeton University
 * June 7, 2021
 */

#ifndef __COMPRESSO_HXX__
#define __COMPRESSO_HXX__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "cc3d.hpp"

namespace compresso {

#define DEFAULT_CONNECTIVITY 4

// little endian serialization of integers to chars
// returns bytes written
inline size_t itoc(uint8_t x, std::vector<unsigned char> &buf, size_t idx) {
	buf[idx] = x;
	return 1;
}

inline size_t itoc(uint16_t x, std::vector<unsigned char> &buf, size_t idx) {
	buf[idx + 0] = x & 0xFF;
	buf[idx + 1] = (x >> 8) & 0xFF;
	return 2;
}

inline size_t itoc(uint32_t x, std::vector<unsigned char> &buf, size_t idx) {
	buf[idx + 0] = x & 0xFF;
	buf[idx + 1] = (x >> 8) & 0xFF;
	buf[idx + 2] = (x >> 16) & 0xFF;
	buf[idx + 3] = (x >> 24) & 0xFF;
	return 4;
}

inline size_t itoc(uint64_t x, std::vector<unsigned char> &buf, size_t idx) {
	buf[idx + 0] = x & 0xFF;
	buf[idx + 1] = (x >> 8) & 0xFF;
	buf[idx + 2] = (x >> 16) & 0xFF;
	buf[idx + 3] = (x >> 24) & 0xFF;
	buf[idx + 4] = (x >> 32) & 0xFF;
	buf[idx + 5] = (x >> 40) & 0xFF;
	buf[idx + 6] = (x >> 48) & 0xFF;
	buf[idx + 7] = (x >> 56) & 0xFF;
	return 8;
}

template <typename T>
T ctoi(unsigned char* buf, size_t idx = 0);

template <>
uint64_t ctoi(unsigned char* buf, size_t idx) {
	uint64_t x = 0;
	x += static_cast<uint64_t>(buf[idx + 0]) << 0;
	x += static_cast<uint64_t>(buf[idx + 1]) << 8;
	x += static_cast<uint64_t>(buf[idx + 2]) << 16;
	x += static_cast<uint64_t>(buf[idx + 3]) << 24;
	x += static_cast<uint64_t>(buf[idx + 4]) << 32;
	x += static_cast<uint64_t>(buf[idx + 5]) << 40;
	x += static_cast<uint64_t>(buf[idx + 6]) << 48;
	x += static_cast<uint64_t>(buf[idx + 7]) << 56;
	return x;
}

template <>
uint32_t ctoi(unsigned char* buf, size_t idx) {
	uint32_t x = 0;
	x += static_cast<uint32_t>(buf[idx + 0]) << 0;
	x += static_cast<uint32_t>(buf[idx + 1]) << 8;
	x += static_cast<uint32_t>(buf[idx + 2]) << 16;
	x += static_cast<uint32_t>(buf[idx + 3]) << 24;
	return x;
}

template <>
uint16_t ctoi(unsigned char* buf, size_t idx) {
	uint16_t x = 0;
	x += static_cast<uint16_t>(buf[idx + 0]) << 0;
	x += static_cast<uint16_t>(buf[idx + 1]) << 8;
	return x;
}

template <>
uint8_t ctoi(unsigned char* buf, size_t idx) {
	return static_cast<uint8_t>(buf[idx]);
}


/* Header: 
 *   'cpso'            : magic number (4 bytes)
 *   format version    : unsigned integer (1 byte) 
 *   data width        : unsigned integer (1 byte) (1: uint8, ... 8: uint64)
 *   sx, sy, sz        : size of each dimension (2 bytes x3)
 *   xstep,ystep,zstep : size of each grid (1 byte x 3) (typical values: 4, 8)
 *   id_size          : number of uniq labels (u64) (could be one per voxel)
 *   value_size       : number of values (u32)
 *   location_size    : number of locations (u64)
 *   connectivity     : CCL algorithm 4 or 6
 */
struct CompressoHeader {
public:
	static constexpr size_t header_size{36};

	static constexpr char magic[4]{ 'c', 'p', 's', 'o' }; 
	static constexpr uint8_t format_version{0};
	uint8_t data_width; // label width in bits
	uint16_t sx;
	uint16_t sy;
	uint16_t sz;
	uint8_t xstep; // 4 bits each to x and y (we only use 4 and 8 anyway)
	uint8_t ystep; // 4 bits each to x and y (we only use 4 and 8 anyway)
	uint8_t zstep; // 4 bits each to x and y (we only use 4 and 8 anyway)
	uint64_t id_size; // label per connected component 
	uint32_t value_size; // boundary encodings (less than size / 16 or size / 64)
	uint64_t location_size; // instructions to remap boundaries
	uint8_t connectivity; // 4 or 6 connected CLL algorithm (almost always 4)

	CompressoHeader() :
		data_width(8), 
		sx(1), sy(1), sz(1), 
		xstep(8), ystep(8), zstep(1),
		id_size(0), value_size(0), location_size(0),
		connectivity(4)
	{}

	CompressoHeader(
		const uint8_t _data_width,
		const uint16_t _sx, const uint16_t _sy, const uint16_t _sz,
		const uint8_t _xstep = 4, const uint8_t _ystep = 4, const uint8_t _zstep = 1,
		const uint64_t _id_size = 0, const uint32_t _value_size = 0, 
		const uint64_t _location_size = 0, const uint8_t _connectivity = 4
	) : 
		data_width(_data_width), 
		sx(_sx), sy(_sy), sz(_sz), 
		xstep(_xstep), ystep(_ystep), zstep(_zstep),
		id_size(_id_size), value_size(_value_size), location_size(_location_size),
		connectivity(_connectivity)
	{}

	CompressoHeader(unsigned char* buf) {
		bool valid_magic = (buf[0] == 'c' && buf[1] == 'p' && buf[2] == 's' && buf[3] == 'o');
		uint8_t format_version = buf[4];

		if (!valid_magic || format_version != 0) {
			throw std::runtime_error("compresso: Data stream is not valid. Unable to decompress.");
		}

		data_width = ctoi<uint8_t>(buf, 5);
		sx = ctoi<uint16_t>(buf, 6); 
		sy = ctoi<uint16_t>(buf, 8); 
		sz = ctoi<uint16_t>(buf, 10);
		xstep = ctoi<uint8_t>(buf, 12); 
		ystep = ctoi<uint8_t>(buf, 13);
		zstep = ctoi<uint8_t>(buf, 14);
		id_size = ctoi<uint64_t>(buf, 15);
		value_size = ctoi<uint32_t>(buf, 23);
		location_size = ctoi<uint64_t>(buf, 27);
		connectivity = ctoi<uint8_t>(buf, 35);

		if (data_width != 1 && data_width != 2 && data_width != 4 && data_width != 8) {
			std::string err = "compresso: Invalid data width in stream. Unable to decompress. Got: ";
			err += std::to_string(data_width);
			throw std::runtime_error(err);
		}
		if (connectivity != 4 && connectivity != 6) {
			std::string err = "compresso: Invalid connectivity in stream. Unable to decompress. Got: ";
			err += std::to_string(connectivity);
			throw std::runtime_error(err);	
		}
	}

	size_t tochars(std::vector<unsigned char> &buf, size_t idx = 0) const {
		if ((idx + CompressoHeader::header_size) > buf.size()) {
			throw std::runtime_error("compresso: Unable to write past end of buffer.");
		}

		size_t i = idx;
		for (int j = 0; j < 4; j++, i++) {
			buf[i] = magic[j];
		}

		i += itoc(format_version, buf, i);
		i += itoc(data_width, buf, i);
		i += itoc(sx, buf, i);
		i += itoc(sy, buf, i);
		i += itoc(sz, buf, i);
		i += itoc(xstep, buf, i);
		i += itoc(ystep, buf, i);
		i += itoc(zstep, buf, i);
		i += itoc(id_size, buf, i);
		i += itoc(value_size, buf, i);
		i += itoc(location_size, buf, i);
		i += itoc(connectivity, buf, i);

		return i - idx;
	}

	static CompressoHeader fromchars(unsigned char* buf) {
		return CompressoHeader(buf);
	}
};

template <typename LABEL>
bool* extract_boundaries(
	LABEL *data, 
	const size_t sx, const size_t sy, const size_t sz,
	const size_t connectivity
) {
	const size_t sxy = sx * sy;
	const size_t voxels = sxy * sz;
	bool *boundaries = new bool[voxels]();

	for (size_t z = 0; z < sz; z++) {
		for (size_t y = 0; y < sy; y++) {
			for (size_t x = 0; x < sx; x++) {
				size_t loc = x + sx * y + sxy * z;

				if (x < sx - 1 && data[loc] != data[loc + 1]) { 
					boundaries[loc] = true;
				}
				else if (y < sy - 1 && data[loc] != data[loc + sx]) {
					boundaries[loc] = true;
				}
				else if (connectivity == 6 && z < sz - 1 && data[loc] != data[loc + sxy]) {
					boundaries[loc] = true;	
				}
			}
		}
	}

	return boundaries;
}

template <typename T>
std::vector<T> component_map(
		uint32_t *components, T *labels, 
		const size_t sx, const size_t sy, const size_t sz,
		const size_t num_components
) {
	const size_t sxy = sx * sy;
	const size_t voxels = sxy * sz;

	std::vector<T> ids(num_components);

	for (size_t i = 0; i < voxels; i++) {
		if (components[i] > 0) {
			ids[components[i] - 1] = labels[i];
		}
	}

	return ids;
}

template <typename T>
std::vector<T> encode_boundaries(
		bool *boundaries, 
		const size_t sx, const size_t sy, const size_t sz, 
		const size_t xstep, const size_t ystep, const size_t zstep
) {

	const size_t sxy = sx * sy;

	const size_t nz = (sz + zstep - 1) / zstep; // round up
	const size_t ny = (sy + ystep - 1) / ystep; // round up
	const size_t nx = (sx + xstep - 1) / xstep; // round up
	const size_t nblocks = nz * ny * nx;

	std::vector<T> boundary_data(nblocks);
	
	size_t xblock, yblock, zblock;
	size_t xoffset, yoffset, zoffset;

	// all these divisions can be replaced by plus/minus
	for (size_t z = 0; z < sz; z++) {
		zblock = z / zstep;
		zoffset = z % zstep;
		for (size_t y = 0; y < sy; y++) {
			yblock = y / ystep;
			yoffset = y % ystep;
			for (size_t x = 0; x < sx; x++) {
				size_t loc = x + sx * y + sxy * z;

				if (!boundaries[loc]) { 
					continue; 
				}

				xblock = x / xstep;
				xoffset = x % xstep;

				size_t block = xblock + nx * yblock + (ny * nx) * zblock;
				size_t offset = xoffset + xstep * yoffset + (ystep * xstep) * zoffset;

				boundary_data[block] += (static_cast<T>(1) << offset);
			}
		}
	}

	return boundary_data;    
}

template <typename T>
std::vector<T> encode_indeterminate_locations(
		bool* boundaries, T* labels, 
		const size_t sx, const size_t sy, const size_t sz,
		const size_t connectivity
) {
	const size_t sxy = sx * sy;
	std::vector<T> locations;
	locations.reserve(sx * sy * sz / 10);

	for (size_t z = 0; z < sz; z++) {
		for (size_t y = 0; y < sy; y++) {
			for (size_t x = 0; x < sx; x++) {
				size_t loc = x + sx * y + sxy * z;
				
				if (!boundaries[loc]) { 
					continue; 
				}
				else if (x > 0 && !boundaries[loc - 1]) {
					continue; 
				}
				else if (y > 0 && !boundaries[loc - sx]) {
					continue;
				}
				else if (connectivity == 6 && z > 0 && !boundaries[loc - sxy]) {
					continue;
				}
				
				size_t left = loc - 1;
				size_t right = loc + 1;
				size_t up = loc - sx;
				size_t down = loc + sx;
				size_t heaven = loc - sxy;
				size_t hell = loc + sxy;

				// see if any of the immediate neighbors are candidates
				if (x > 0 && !boundaries[left] && (labels[left] == labels[loc])) {
					locations.push_back(0);
				}
				else if (x < sx - 1 && !boundaries[right] && (labels[right] == labels[loc])) {
					locations.push_back(1);
				}
				else if (y > 0 && !boundaries[up] && (labels[up] == labels[loc])) {
					locations.push_back(2);
				}
				else if (y < sy - 1 && !boundaries[down] && (labels[down] == labels[loc])) {
					locations.push_back(3);
				}
				else if (z > 0 && !boundaries[heaven] && (labels[heaven] == labels[loc])) {
					locations.push_back(4);
				}
				else if (z < sz - 1 && !boundaries[hell] && (labels[hell] == labels[loc])) {
					locations.push_back(5);
				}
				else if (labels[loc] > std::numeric_limits<T>::max() - 7) {
					locations.push_back(6);
					locations.push_back(labels[loc]);
				}
				else {
					locations.push_back(labels[loc] + 7);
				}
			}
		}
	}

	return locations;
}

template <typename T>
std::vector<T> unique(const std::vector<T> &data) {
	std::vector<T> values;

	if (data.size() == 0) {
		return values;
	}

	std::set<T> hash_map;
	T last = data[0];
	hash_map.insert(data[0]);
	values.push_back(data[0]);

	for (size_t iv = 1; iv < data.size(); iv++) {
		if (data[iv] == last) {
			continue;
		}

		bool inserted = hash_map.insert(data[iv]).second;
		if (inserted) {
			values.push_back(data[iv]);
		}
		last = data[iv];
	}
	sort(values.begin(), values.end());
	return values;
}

template <typename T>
void renumber_boundary_data(const std::vector<T>& window_values, std::vector<T> &windows) {
	if (windows.size() == 0) {
		return;
	}

	std::unordered_map<T, T> mapping;
	for (size_t iv = 0; iv < window_values.size(); iv++) {
		mapping[window_values[iv]] = iv;
	}

	T last = windows[0];
	windows[0] = mapping[windows[0]];
	T last_remap = windows[0];

	for (size_t iv = 1; iv < windows.size(); iv++) {
		if (windows[iv] == last) {
			windows[iv] = last_remap;
			continue;
		}

		last_remap = mapping[windows[iv]];
		last = windows[iv];
		windows[iv] = last_remap;
	}
}

template <>
void renumber_boundary_data(const std::vector<uint16_t>& window_values, std::vector<uint16_t> &windows) {
	if (windows.size() == 0) {
		return;
	}

	std::vector<uint16_t> mapping(pow(2,16));
	for (size_t iv = 0; iv < window_values.size(); iv++) {
		mapping[window_values[iv]] = iv;
	}
	for (size_t iv = 0; iv < windows.size(); iv++) {
		windows[iv] = mapping[windows[iv]];
	}
}

template <typename T>
std::vector<T> run_length_encode_windows(const std::vector<T> &windows) {
	std::vector<T> rle_windows;
	rle_windows.reserve(windows.size() / 4);

	size_t zero_run = 0;
	size_t max_run = std::numeric_limits<T>::max() / 2;

	for (size_t i = 0; i < windows.size(); i++) {
		if (windows[i] > max_run) {
			throw std::runtime_error("compresso: Unable to RLE encode. Too many windows. Use 64-bit steps e.g. (8,8,1) instead.");
		}
		else if (windows[i] == 0) {
			zero_run++;
			if (zero_run < max_run) {
				continue;
			}
		}
		
		if (zero_run) {
			rle_windows.push_back((zero_run << 1) | 1);
			zero_run = 0;
		}
		rle_windows.push_back(windows[i] << 1);
	}

	if (zero_run) {
		rle_windows.push_back((zero_run << 1) | 1);
	}

	return rle_windows;
}

template <typename WINDOW>
std::vector<WINDOW> run_length_decode_windows(
	const std::vector<WINDOW> &rle_windows, const size_t nblocks
) {
	std::vector<WINDOW> windows(nblocks);

	WINDOW block = 0;
	size_t index = 0;
	const size_t window_size = rle_windows.size();

	for (size_t i = 0; i < window_size; i++) {
		block = rle_windows[i];
		if (block & 1) {
			index += (block >> 1);
		}
		else {
			windows[index] = block >> 1;
			index++;
		}
	}

	return windows;
}

template <typename LABEL, typename WINDOW>
void write_compressed_stream(
	std::vector<unsigned char> &compressed_data,
	const CompressoHeader &header, 
	const std::vector<LABEL> &ids, 
	const std::vector<WINDOW> &window_values, 
	const std::vector<LABEL> &locations,
	const std::vector<WINDOW> &windows
) {
	size_t idx = header.tochars(compressed_data, 0);
	for (size_t i = 0 ; i < ids.size(); i++) {
		idx += itoc(ids[i], compressed_data, idx);
	}
	for (size_t i = 0 ; i < window_values.size(); i++) {
		idx += itoc(window_values[i], compressed_data, idx);
	}
	for (size_t i = 0 ; i < locations.size(); i++) {
		idx += itoc(locations[i], compressed_data, idx);
	}
	for (size_t i = 0 ; i < windows.size(); i++) {
		idx += itoc(windows[i], compressed_data, idx);
	}
}

template <typename LABEL, typename WINDOW>
std::vector<unsigned char> compress_helper(
	LABEL* labels, 
	const size_t sx, const size_t sy, const size_t sz,
	const size_t xstep, const size_t ystep, const size_t zstep,
	const size_t connectivity, bool* boundaries, const std::vector<LABEL>& ids
) {

	std::vector<WINDOW> windows = encode_boundaries<WINDOW>(boundaries, sx, sy, sz, xstep, ystep, zstep);
	std::vector<LABEL> locations = encode_indeterminate_locations<LABEL>(
		boundaries, labels, sx, sy, sz, connectivity
	);
	delete[] boundaries;

	std::vector<WINDOW> window_values = unique<WINDOW>(windows);
	renumber_boundary_data(window_values, windows);
	windows = run_length_encode_windows<WINDOW>(windows);

	size_t num_out_bytes = (
		CompressoHeader::header_size 
		+ (ids.size() * sizeof(LABEL))
		+ (window_values.size() * sizeof(WINDOW))
		+ (locations.size() * sizeof(LABEL))
		+ (windows.size() * sizeof(WINDOW))
	);
	std::vector<unsigned char> compressed_data(num_out_bytes);

	CompressoHeader header(
		/*data_width=*/sizeof(LABEL), 
		/*sx=*/sx, /*sy=*/sy, /*sz=*/sz,
		/*xstep=*/xstep, /*ystep=*/ystep, /*zstep=*/zstep,
		/*id_size=*/ids.size(), 
		/*value_size=*/window_values.size(), 
		/*location_size=*/locations.size(),
		/*connectivity=*/connectivity
	);

	write_compressed_stream<LABEL, WINDOW>(
		compressed_data, header, ids, 
		window_values, locations, windows
	);

	return compressed_data;
}

std::vector<unsigned char> zero_data_stream(	
	const size_t sx, const size_t sy, const size_t sz,
	const size_t xstep, const size_t ystep, const size_t zstep,
	const size_t data_width, const size_t connectivity
) {
	std::vector<unsigned char> compressed_data(CompressoHeader::header_size);
	
	CompressoHeader empty_header(
		/*data_width=*/data_width, 
		/*sx=*/sx, /*sy=*/sy, /*sz=*/sz,
		/*xstep=*/xstep, /*ystep=*/ystep, /*zstep=*/zstep,
		/*id_size=*/0, 
		/*value_size=*/0, 
		/*location_size=*/0,
		/*connectivity*/connectivity
	);
	empty_header.tochars(compressed_data);
	return compressed_data;
}

/* compress
 *
 * Convert 3D integer array data into a compresso encoded byte stream.
 * Array is expected to be in Fortran order.
 *
 * Parameters:
 *  data: pointer to 3D integer segmentation image 
 *  sx, sy, sz: axial dimension sizes
 *  xstep, ystep, zstep: (optional) picks the size of the 
 *      compresso grid. 4x4x1 or 8x8x1 are acceptable sizes.
 *
 * Returns: vector<char>
 */
template <typename T>
std::vector<unsigned char> compress(
	T* labels, 
	const size_t sx, const size_t sy, const size_t sz,
	const size_t xstep = 4, const size_t ystep = 4, const size_t zstep = 1,
	const size_t connectivity = 4
) {

	if (sx * sy * sz == 0) {
		return zero_data_stream(sx, sy, sz, xstep, ystep, zstep, sizeof(T), connectivity);
	}

	if (xstep * ystep * zstep > 64) {
		throw std::runtime_error("compresso: Unable to encode blocks larger than 64 voxels.");
	}
	else if (xstep * ystep * zstep == 0) {
		throw std::runtime_error("compresso: Unable to encode using zero step sizes.");	
	}

	bool *boundaries = extract_boundaries<T>(labels, sx, sy, sz, connectivity);
	size_t num_components = 0;
	uint32_t *components = cc3d::connected_components<uint32_t>(
		boundaries, sx, sy, sz, 
		/*connectivity=*/connectivity, num_components
	);
	
	std::vector<T> ids = component_map<T>(components, labels, sx, sy, sz, num_components);
	delete[] components;

	// can use a more efficient window size
	// if the grid size is small enough. 
	// specifically, we're talking about
	// 4x4x1 step size
	if (xstep * ystep * zstep <= 8) {
		return compress_helper<T, uint8_t>(
			labels, 
			sx, sy, sz, 
			xstep, ystep, zstep, connectivity,
			boundaries, ids
		);
	}
	else if (xstep * ystep * zstep <= 16) {
		return compress_helper<T, uint16_t>(
			labels, 
			sx, sy, sz, 
			xstep, ystep, zstep, connectivity,
			boundaries, ids
		);
	}
	else if (xstep * ystep * zstep <= 32) { // 4x4x2 for example
		return compress_helper<T, uint32_t>(
			labels, 
			sx, sy, sz, 
			xstep, ystep, zstep, connectivity,
			boundaries, ids
		);
	}
	else { // for 8x8x1 step size
		return compress_helper<T, uint64_t>(
			labels, 
			sx, sy, sz, 
			xstep, ystep, zstep, connectivity,
			boundaries, ids
		);
	}
}

/* DECOMPRESS STARTS HERE */

template <typename LABEL, typename WINDOW>
bool* decode_boundaries(
	const std::vector<WINDOW> &windows, const std::vector<WINDOW> &window_values, 
	const size_t sx, const size_t sy, const size_t sz,
	const size_t xstep, const size_t ystep, const size_t zstep
) {

	const size_t sxy = sx * sy;
	const size_t voxels = sx * sy * sz;

	const size_t nx = (sx + xstep - 1) / xstep; // round up
	const size_t ny = (sy + ystep - 1) / ystep; // round up

	// check for power of two
	const bool xstep_pot = (xstep != 0) && ((xstep & (xstep - 1)) == 0);
	const int xshift = std::log2(xstep); // must use log2 here, not lg/lg2 to avoid fp errors

	bool* boundaries = new bool[voxels]();

	if (window_values.size() == 0) {
		return boundaries;
	}

	size_t xblock, yblock, zblock;
	size_t xoffset, yoffset, zoffset;

	for (size_t z = 0; z < sz; z++) {
		zblock = nx * ny * (z / zstep);
		zoffset = xstep * ystep * (z % zstep);
		for (size_t y = 0; y < sy; y++) {
			yblock = nx * (y / ystep);
			yoffset = xstep * (y % ystep);

			if (xstep_pot) {
				for (size_t x = 0; x < sx; x++) {
					size_t iv = x + sx * y + sxy * z;

					xblock = x >> xshift; // x / xstep
					xoffset = x & ((1 << xshift) - 1); // x % xstep
					
					size_t block = xblock + yblock + zblock;
					size_t offset = xoffset + yoffset + zoffset;

					WINDOW value = window_values[windows[block]];
					boundaries[iv] = (value >> offset) & 0b1;
				}				
			}
			else {
				for (size_t x = 0; x < sx; x++) {
					size_t iv = x + sx * y + sxy * z;
					xblock = x / xstep;
					xoffset = x % xstep;
					
					size_t block = xblock + yblock + zblock;
					size_t offset = xoffset + yoffset + zoffset;

					WINDOW value = window_values[windows[block]];
					boundaries[iv] = (value >> offset) & 0b1;
				}
			}
		}
	}

	return boundaries;
}

template <typename LABEL>
void decode_nonboundary_labels(
		uint32_t *components, const std::vector<LABEL> &ids, 
		const size_t sx, const size_t sy, const size_t sz,
		LABEL* output
) {
	const size_t voxels = sx * sy * sz;
	for (size_t i = 0; i < voxels; i++) {
		output[i] = ids[components[i]];
	}
}

template <typename LABEL>
void decode_indeterminate_locations(
		bool *boundaries, LABEL *labels, 
		const std::vector<LABEL> &locations, 
		const size_t sx, const size_t sy, const size_t sz,
		const size_t connectivity
) {
	const size_t sxy = sx * sy;

	size_t loc = 0;
	size_t index = 0;

	// go through all coordinates
	for (size_t z = 0; z < sz; z++) {
		for (size_t y = 0; y < sy; y++) {
			for (size_t x = 0; x < sx; x++) {
				loc = x + sx * y + sxy * z;

				if (!boundaries[loc]) {
					continue;
				}
				else if (x > 0 && !boundaries[loc - 1]) {
					labels[loc] = labels[loc - 1];
					continue;
				}
				else if (y > 0 && !boundaries[loc - sx]) {
					labels[loc] = labels[loc - sx];
					continue;
				}
				else if (connectivity == 6 && z > 0 && !boundaries[loc - sxy]) {
					labels[loc] = labels[loc - sxy];
					continue;
				}
				else if (locations.size() == 0) {
					throw std::runtime_error("compresso: unable to decode indeterminate locations. (no locations)");
				}
				
				size_t offset = locations[index];

				if (offset == 0) {
					if (x == 0) {
						throw std::runtime_error("compresso: unable to decode indeterminate locations. (offset 0)");
					}
					labels[loc] = labels[loc - 1];
				}
				else if (offset == 1) {
					if (x >= sx - 1) {
						throw std::runtime_error("compresso: unable to decode indeterminate locations. (offset 1)");
					}
					labels[loc] = labels[loc + 1];
				}
				else if (offset == 2) {
					if (y == 0) {
						throw std::runtime_error("compresso: unable to decode indeterminate locations. (offset 2)");
					}
					labels[loc] = labels[loc - sx];
				}
				else if (offset == 3) {
					if (y >= sy - 1) {
						throw std::runtime_error("compresso: unable to decode indeterminate locations. (offset 3)");
					}
					labels[loc] = labels[loc + sx];
				}
				else if (offset == 4) {
					if (z == 0) {
						throw std::runtime_error("compresso: unable to decode indeterminate locations. (offset 4)");
					}
					labels[loc] = labels[loc - sxy];
				}
				else if (offset == 5) {
					if (z >= sz - 1) {
						throw std::runtime_error("compresso: unable to decode indeterminate locations. (offset 5)");
					}
					labels[loc] = labels[loc + sxy];
				}
				else if (offset == 6) {
					labels[loc] = locations[index + 1];
					index++;
				}
				else {
					labels[loc] = offset - 7;
				}
				index++;
			}
		}
	}
}

template <typename LABEL, typename WINDOW>
LABEL* decompress(unsigned char* buffer, size_t num_bytes, LABEL* output = NULL) {

	if (num_bytes < CompressoHeader::header_size) {
		std::string err = "compresso: Input too small to be a valid stream. Bytes: ";
		err += std::to_string(num_bytes);
		throw std::runtime_error(err);
	}

	const CompressoHeader header(buffer);

	const size_t sx = header.sx;
	const size_t sy = header.sy;
	const size_t sz = header.sz;
	const size_t voxels = sx * sy * sz;
	const size_t xstep = header.xstep;
	const size_t ystep = header.ystep;
	const size_t zstep = header.zstep;

	if (sx * sy * sz == 0) {
		return NULL;
	}

	const size_t nx = (sx + xstep - 1) / xstep; // round up
	const size_t ny = (sy + ystep - 1) / ystep; // round up
	const size_t nz = (sz + zstep - 1) / zstep; // round up
	const size_t nblocks = nz * ny * nx;

	size_t window_bytes = (
		num_bytes 
			- CompressoHeader::header_size
			- (header.id_size * sizeof(LABEL))  
			- (header.value_size * sizeof(WINDOW))
			- (header.location_size * sizeof(LABEL))
	);
	size_t num_condensed_windows = window_bytes / sizeof(WINDOW);

	// allocate memory for all arrays
	std::vector<LABEL> ids(header.id_size + 1); // +1 to allow vectorized mapping w/ no if statement guarding zero
	std::vector<WINDOW> window_values(header.value_size);
	std::vector<LABEL> locations(header.location_size);
	std::vector<WINDOW> windows(num_condensed_windows);

	size_t iv = CompressoHeader::header_size;
	for (size_t ix = 0; ix < ids.size() - 1; ix++, iv += sizeof(LABEL)) {
		ids[ix + 1] = ctoi<LABEL>(buffer, iv);
	}
	for (size_t ix = 0; ix < window_values.size(); ix++, iv += sizeof(WINDOW)) {
		window_values[ix] = ctoi<WINDOW>(buffer, iv);
	}
	for (size_t ix = 0; ix < locations.size(); ix++, iv += sizeof(LABEL)) {
		locations[ix] = ctoi<LABEL>(buffer, iv);
	}
	for (size_t ix = 0; ix < num_condensed_windows; ix++, iv += sizeof(WINDOW)) {
		windows[ix] = ctoi<WINDOW>(buffer, iv);
	}

	windows = run_length_decode_windows<WINDOW>(windows, nblocks);

	bool* boundaries = decode_boundaries<WINDOW>(
		windows, window_values, 
		sx, sy, sz, 
		xstep, ystep, zstep
	);
	windows = std::vector<WINDOW>();
	window_values = std::vector<WINDOW>();

	uint32_t* components = cc3d::connected_components<uint32_t>(
		boundaries, sx, sy, sz, header.connectivity
	);

	if (output == NULL) {
		output = new LABEL[voxels]();
	}

	decode_nonboundary_labels(components, ids, sx, sy, sz, output);
	delete[] components;
	ids = std::vector<LABEL>();

	decode_indeterminate_locations<LABEL>(
		boundaries, output, locations, 
		sx, sy, sz,
		header.connectivity
	);

	delete[] boundaries;

	return output;
}

template <>
void* decompress<void,void>(unsigned char* buffer, size_t num_bytes, void* output) {
	CompressoHeader header(buffer);

	bool window8 = (
		static_cast<int>(header.xstep) * static_cast<int>(header.ystep) * static_cast<int>(header.zstep) <= 8
	);
	bool window16 = (
		static_cast<int>(header.xstep) * static_cast<int>(header.ystep) * static_cast<int>(header.zstep) <= 16
	);
	bool window32 = (
		static_cast<int>(header.xstep) * static_cast<int>(header.ystep) * static_cast<int>(header.zstep) <= 32
	);

	if (header.data_width == 1) {
		if (window8) {
			return reinterpret_cast<void*>(
				decompress<uint8_t,uint8_t>(
					buffer, num_bytes, reinterpret_cast<uint8_t*>(output)
				)
			);
		}
		else if (window16) {
			return reinterpret_cast<void*>(
				decompress<uint8_t,uint16_t>(
					buffer, num_bytes, reinterpret_cast<uint8_t*>(output)
				)
			);
		}
		else if (window32) {
			return reinterpret_cast<void*>(
				decompress<uint8_t,uint32_t>(
					buffer, num_bytes, reinterpret_cast<uint8_t*>(output)
				)
			);
		}
		else {
			return reinterpret_cast<void*>(
				decompress<uint8_t, uint64_t>(
					buffer, num_bytes, reinterpret_cast<uint8_t*>(output)
				)
			);			
		}
	}
	else if (header.data_width == 2) {
		if (window8) {
			return reinterpret_cast<void*>(
				decompress<uint16_t,uint8_t>(
					buffer, num_bytes, reinterpret_cast<uint16_t*>(output)
				)
			);
		}
		else if (window16) {
			return reinterpret_cast<void*>(
				decompress<uint16_t,uint16_t>(
					buffer, num_bytes, reinterpret_cast<uint16_t*>(output)
				)
			);
		}
		else if (window32) {
			return reinterpret_cast<void*>(
				decompress<uint16_t,uint32_t>(
					buffer, num_bytes, reinterpret_cast<uint16_t*>(output)
				)
			);
		}
		else {
			return reinterpret_cast<void*>(
				decompress<uint16_t, uint64_t>(
					buffer, num_bytes, reinterpret_cast<uint16_t*>(output)
				)
			);			
		}
	}
	else if (header.data_width == 4) {
		if (window8) {
			return reinterpret_cast<void*>(
				decompress<uint32_t,uint8_t>(
					buffer, num_bytes, reinterpret_cast<uint32_t*>(output)
				)
			);
		}
		else if (window16) {
			return reinterpret_cast<void*>(
				decompress<uint32_t,uint16_t>(
					buffer, num_bytes, reinterpret_cast<uint32_t*>(output)
				)
			);		
		}
		else if (window32) {
			return reinterpret_cast<void*>(
				decompress<uint32_t,uint32_t>(
					buffer, num_bytes, reinterpret_cast<uint32_t*>(output)
				)
			);
		}
		else {
			return reinterpret_cast<void*>(
				decompress<uint32_t, uint64_t>(
					buffer, num_bytes, reinterpret_cast<uint32_t*>(output)
				)
			);			
		}
	}
	else if (header.data_width == 8) {
		if (window8) {
			return reinterpret_cast<void*>(
				decompress<uint64_t,uint8_t>(
					buffer, num_bytes, reinterpret_cast<uint64_t*>(output)
				)
			);
		}
		else if (window16) {
			return reinterpret_cast<void*>(
				decompress<uint64_t,uint16_t>(
					buffer, num_bytes, reinterpret_cast<uint64_t*>(output)
				)
			);		
		}
		else if (window32) {
			return reinterpret_cast<void*>(
				decompress<uint64_t,uint32_t>(
					buffer, num_bytes, reinterpret_cast<uint64_t*>(output)
				)
			);
		}
		else {
			return reinterpret_cast<void*>(
				decompress<uint64_t, uint64_t>(
					buffer, num_bytes, reinterpret_cast<uint64_t*>(output)
				)
			);			
		}
	}
	else {
		std::string err = "compresso: Invalid data width: ";
		err += std::to_string(header.data_width);
		throw std::runtime_error(err);
	}
}

};

namespace pycompresso {

static constexpr size_t COMPRESSO_HEADER_SIZE{compresso::CompressoHeader::header_size};

std::vector<unsigned char> cpp_zero_data_stream(	
	const size_t sx, const size_t sy, const size_t sz,
	const size_t xstep, const size_t ystep, const size_t zstep,
	const size_t data_width, const size_t connectivity
) {
	return compresso::zero_data_stream(sx, sy, sz, xstep, ystep, zstep, data_width, connectivity);
}

template <typename T>
std::vector<unsigned char> cpp_compress(
	T* labels, 
	const size_t sx, const size_t sy, const size_t sz,
	const size_t xstep = 4, const size_t ystep = 4, const size_t zstep = 1,
	const size_t connectivity = 4
) {

	return compresso::compress<T>(labels, sx, sy, sz, xstep, ystep, zstep, connectivity);
}

void* cpp_decompress(unsigned char* buffer, size_t num_bytes, void* output) {
	return compresso::decompress<void,void>(buffer, num_bytes, output);
}

};

#endif