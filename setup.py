import setuptools
import sys

import numpy as np

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
      include_dirs=[ np.get_include() ],
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
