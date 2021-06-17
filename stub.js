// Used when compiling wasm module.
mergeInto(LibraryManager.library, {
  _neuroglancer_compresso_decompress: function() {
    alert(arguments);
  },
});