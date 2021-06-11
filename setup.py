import setuptools

import numpy as np

setuptools.setup(
  setup_requires=['pbr', 'numpy'],
  install_requires=['numpy'],
  ext_modules=[
    setuptools.Extension(
      'compresso',
      include_dirs=[ np.get_include() ],
      sources=['compresso.cpp'],
      extra_compile_args=['-O3', '-std=c++11'],
      language='c++'
    )
  ],
  pbr=True)
