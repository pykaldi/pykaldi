#!/usr/bin/env python
#
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup configuration."""

import os
import platform
import sys

try:
  import setuptools
except ImportError:
  from ez_setup import use_setuptools
  use_setuptools()
  import setuptools

py_version = platform.python_version_tuple()
if py_version < ('2', '7') or py_version[0] == '3' and py_version < ('3', '4'):
  raise RuntimeError('Python version 2.7 or 3.4+ required')

user_home = os.getenv('HOME')

setuptools.setup(
    name='pykaldi',
    version='0.0.1',
    description='Kaldi Python Wrapper',
    url='https://github.com/usc-sail/pykaldi',
    author='VRMP',
    author_email='victorrm@usc.edu',
    install_requires=['enum34;python_version<"3.4"'],
    ext_modules=[
        setuptools.Extension(
            'kaldi-utils', [
                # CLIF-generated sources
                'kaldi/src/base/python/kaldi-utils.cc',
                'kaldi/src/base/python/kaldi-utils_init.cc',
                # 'clif_runtime',
                user_home + '/opt/clif/python/runtime.cc',
                user_home + '/opt/clif/python/slots.cc',
                user_home + '/opt/clif/python/types.cc',
                ],
            include_dirs=[
                # Path to clif runtime headers and example cc lib headers
                user_home + '/opt',
                ],
            extra_compile_args=['-std=c++11'],
            library_dirs=['./'],
            libraries=[''],
            extra_objects=[],
            ),
        ],
    )