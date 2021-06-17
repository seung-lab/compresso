// Used when compiling wasm module.
mergeInto(LibraryManager.library, {
  compresso_receive_decoded_image: function() {
    alert(arguments);
  },
});