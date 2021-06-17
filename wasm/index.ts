/**
 * @license
 * Copyright 2019 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * Modified from  
 * https://github.com/google/neuroglancer/blob/8432f531c4d8eb421556ec36926a29d9064c2d3c/src/neuroglancer/mesh/draco/index.ts
 * by William Silversmith, 2021
 */

import compressoWasmUrl from './compresso.wasm';

declare const WebAssembly: any;

const memory = new WebAssembly.Memory({initial: 1});
let heap8: Uint8Array;
function updateHeapViews() {
  heap8 = new Uint8Array(memory.buffer);
}
updateHeapViews();
var heap32 = new Uint32Array(memory.buffer);
var DYNAMIC_BASE = 38592, DYNAMICTOP_PTR = 5568;
heap32[DYNAMICTOP_PTR >> 2] = DYNAMIC_BASE;
function abort() {
  throw 'abort';
}
function alignUp(x: number, multiple: number) {
  if (x % multiple > 0) {
    x += multiple - x % multiple
  }
  return x
}
function emscripten_realloc_buffer(size: number) {
  var PAGE_MULTIPLE = 65536;
  size = alignUp(size, PAGE_MULTIPLE);
  var oldSize = heap8.byteLength;
  try {
    var result = memory.grow((size - oldSize) / 65536);
    if (result !== (-1 | 0)) {
      return true;
    } else {
      return false;
    }
  } catch (e) {
    return false;
  }
}

let decodeResult: TypedArray|Error|undefined = undefined;

const imports = {
  env: {
    memory: memory,
    table: new WebAssembly.Table({'initial': 368, 'maximum': 368, 'element': 'anyfunc'}),
    __memory_base: 1024,
    __table_base: 0,
    _compresso_receive_decoded_image: function(
        image_ptr: number, 
        sx: number, sy: number, sz: number, 
        data_width: number
    ) {
      const voxels = sx * sy * sz;
      let image : TypedArray;
      if (data_width === 1) {
        image = new Uint8Array(memory.buffer, image_ptr, voxels);
      }
      else if (data_width === 2) {
        image = new Uint16Array(memory.buffer, image_ptr, voxels);
      }
      else if (data_width === 4) {
        image = new Uint32Array(memory.buffer, image_ptr, voxels);
      }
      else if (data_width === 8) {
        image = new BigUint64Array(memory.buffer, image_ptr, voxels);
      }
      else {
        decodeResult = new Error(`data_width must be 1, 2, 4, or 8. Got: ${data_width}`);
        return;
      }

      decodeResult = image;
    },
    _emscripten_memcpy_big: function(d: number, s: number, n: number) {
      heap8.set(heap8.subarray(s, s + n), d);
    },
    _emscripten_get_heap_size: function() {
      return heap8.length;
    },
    DYNAMICTOP_PTR: DYNAMICTOP_PTR,
    _abort: abort,
    abort: abort,
    abortOnCannotGrowMemory: abort,
    ___cxa_pure_virtual: abort,
    _llvm_trap: abort,
    ___setErrNo: abort,
    _emscripten_resize_heap: function(requestedSize: number) {
      var oldSize = heap8.length;
      var PAGE_MULTIPLE = 65536;
      var LIMIT = 2147483648 - PAGE_MULTIPLE;
      if (requestedSize > LIMIT) {
        return false
      }
      var MIN_TOTAL_MEMORY = 16777216;
      var newSize = Math.max(oldSize, MIN_TOTAL_MEMORY);
      while (newSize < requestedSize) {
        if (newSize <= 536870912) {
          newSize = alignUp(2 * newSize, PAGE_MULTIPLE)
        } else {
          newSize = Math.min(alignUp((3 * newSize + 2147483648) / 4, PAGE_MULTIPLE), LIMIT)
        }
      }
      var replacement = emscripten_realloc_buffer(newSize);
      if (!replacement) {
        return false
      }
      updateHeapViews();
      return true
    },
  },
};

const compressoModulePromise = fetch(compressoWasmUrl)
                               .then(response => response.arrayBuffer())
                               .then(wasmCode => WebAssembly.instantiate(wasmCode, imports));

function imageSize (buffer: Uint8Array) : number {
  const magic = (
       buffer[0] == 'c' && buffer[1] == 'p' 
    && buffer[2] == 's' && buffer[3] == 'o'
  );
  if (!magic) {
    return -1;
  }
  const format = buffer[4];
  if (format !== 0) {
    return -2;
  }

  let u16 = (lower, upper) => { return (lower|0) + (upper << 8) };

  const data_width = buffer[5];
  const sx = u16(buffer[6], buffer[7]);
  const sy = u16(buffer[8], buffer[9]);
  const sz = u16(buffer[9], buffer[10]);

  return sx * sy * sz * data_width;
}

export function decodeCompresso(buffer: Uint8Array) 
  : Promise<TypedArray> 
{
  return compressoModulePromise.then(m => {
    const nbytes = imageSize(buffer);
    if (nbytes < 0) {
      throw new Error(`Failed to decode compresso image. imageSize code: ${nbytes}`);
    }

    const buf_ptr = m.instance.exports._malloc(buffer.byteLength);
    heap8.set(buffer, buf_ptr);

    const image_ptr = m.instance.exports._malloc(nbytes);
    
    const code = m.instance.exports._compresso_decompress(
      buf_ptr, buffer.byteLength, image_ptr
    );
    
    if (code === 0) {
      const r = decodeResult;
      decodeResult = undefined;
      if (r instanceof Error) throw r;
      return r!;
    }
    throw new Error(`Failed to decode compresso image. decoder code: ${code}`);
  });
}