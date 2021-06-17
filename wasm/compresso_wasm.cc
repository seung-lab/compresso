#include <cstdint>
#include <memory>

#include "../compresso.hpp"

namespace {
struct FreeDeleter {
	void operator()(void *p) const { ::free(p); }
};
}

extern "C" {

extern void compresso_receive_decoded_image(
	const void *image,
	const unsigned int sx, const unsigned int sy, const unsigned int sz, 
	const unsigned int data_width
);

int compresso_decompress(
	unsigned char* buf, unsigned int num_bytes, void* out
) {
	std::unique_ptr<unsigned char[], FreeDeleter> input_deleter(buf);

	try {
		compresso::decompress<void,void>(buf, num_bytes, out);	
	}
	catch(std::runtime_error& e) {
		return 1;
	}

	compresso::CompressoHeader header(buf);

	compresso_receive_decoded_image(
		out, 
		header.sx, header.sy, header.sz, 
		header.data_width
	);

	return 0;
}

}