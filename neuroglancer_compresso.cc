#include <cstdint>
#include <memory>

#include "compresso.hpp"

namespace {
struct FreeDeleter {
  void operator()(void *p) const { ::free(p); }
};
}

extern "C" {

int neuroglancer_compresso_decompress(
	unsigned char* buf, unsigned int num_bytes, void* out
) {
  std::unique_ptr<unsigned char[], FreeDeleter> input_deleter(buf);

  try {
  	compresso::decompress<void,void>(buf, num_bytes, out);	
  }
  catch(std::runtime_error& e) {
  	return 1;
  }

  return 0;
}

}