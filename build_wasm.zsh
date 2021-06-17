#!/bin/zsh -xve

compile_options=(
    neuroglancer_compresso.cc
     -O3
     -DNDEBUG
     --js-library stub.js
     --no-entry
     -fno-rtti
     -s FILESYSTEM=0
     -s ALLOW_MEMORY_GROWTH=1 
     -s TOTAL_STACK=32768 -s TOTAL_MEMORY=65536
     -s EXPORTED_FUNCTIONS='["_neuroglancer_compresso_decompress","_malloc"]'
     -s MALLOC=emmalloc
     -s ENVIRONMENT=worker
     -std=c++14
     -o compresso.wasm
)
emcc ${compile_options[@]}
