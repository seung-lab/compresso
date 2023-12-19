import setuptools
import sys

class NumpyImport:
  def __repr__(self):
    import numpy as np

    return np.get_include()

  __fspath__ = __repr__

extra_compile_args = []
if sys.platform == 'win32':
  extra_compile_args += [
    '/std:c++11', '/O2'
  ]
else:
  extra_compile_args += [
    '-std=c++11', '-O3'
  ]

setuptools.setup(
  setup_requires=['pbr', 'numpy'],
  install_requires=['numpy'],
  ext_modules=[
    setuptools.Extension(
      'compresso',
      include_dirs=[ str(NumpyImport()) ],
      sources=['compresso.cpp'],
      extra_compile_args=extra_compile_args,
      language='c++'
    )
  ],
  entry_points={
    "console_scripts": [
      "compresso=compresso_cli:main"
    ],
  },
  pbr=True)
