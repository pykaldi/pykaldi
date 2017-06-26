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

from setuptools import setup, Extension, distutils, Command, find_packages
import setuptools.command.build_ext
import setuptools.command.install
import setuptools.command.develop
import setuptools.command.build_py
import platform
import subprocess
import shutil
import sys
import os


################################################################################
# From: https://github.com/pytorch/pytorch/blob/master/tools/setup_helpers/env.py
################################################################################
def check_env_flag(name):
	return os.getenv(name) in ['ON', '1', 'YES', 'TRUE', 'Y']

################################################################################
# From: https://stackoverflow.com/questions/377017/test-if-executable-exists-in-python
################################################################################
def which(program):
	def is_exe(fpath):
		return os.path.isfile(fpath) and os.access(fpath, os.X_OK)
	fpath, fname = os.path.split(program)
	if fpath:
		if is_exe(program):
			return program
	else:
		for path in os.environ['PATH'].split(os.pathsep):
			path = path.strip('"')
			exe_file = os.path.join(path, program)
			if is_exe(exe_file):
				return exe_file
	return None


################################################################################
# Check variables / find programs
################################################################################
DEBUG = check_env_flag('DEBUG')
PYCLIF = which("pyclif")

if not PYCLIF:
	PYCLIF = os.getenv("PYCLIF")
	if not PYCLIF:
		print("We could not find PYCLIF. Try setting PYCLIF environment variable.")
		sys.exit(1)
		
if "KALDI_DIR" not in os.environ:
	# KALDI = which("kaldi")
	# if not KALDI:
	print("We could not find KALDI. Try setting KALDI_DIR environment variable.")
	sys.exit(1)
	# else:
	# 	KALDI_DIR = os.path.join(KALDI, "..")

cwd = os.path.dirname(os.path.abspath(__file__))
opt = os.path.join(PYCLIF_BIN, "../../..")

if DEBUG:
	print("#"*25)
	print("CWD: {}".format(cwd))
	print("PYCLIF_BIN: {}".format(PYCLIF_BIN))
	print("KALDI_DIR: {}".format(KALDI_DIR))
	print("OPT_DIR: {}".format(opt))
	print("CXX_FLAGS: {}".format(os.getenv("CXX_FLAGS")))
	print("#"*25)

################################################################################
# Workaround setuptools -Wstrict-prototypes warnings
# From: https://github.com/pytorch/pytorch/blob/master/setup.py
################################################################################
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
for key, value in cfg_vars.items():
	if type(value) == str:
			cfg_vars[key] = value.replace("-Wstrict-prototypes", "")

################################################################################
# Custom build commands
################################################################################
class build_deps(Command):
	user_options = []

	def initialize_options(self):
		pass

	def finalize_options(self):
		pass

	def run(self):
		build_all_cmd = ['bash', 'build_all.sh', PYCLIF_BIN, KALDI_DIR]
		if subprocess.call(build_all_cmd) != 0:
			sys.exit(1)

################################################################################
# Configure compile flags
################################################################################
library_dirs = [os.path.join(KALDI_DIR, '/src/lib/')]

include_dirs = [
	cwd,
	# Path to clif runtime headers and example cc lib headers
	opt,
	os.path.join(KALDI_DIR, 'src/'),
	os.path.join(KALDI_DIR, 'tools/openfst/include'),
	os.path.join(KALDI_DIR, 'tools/ATLAS/include')
]

extra_compile_args = [
	'-std=c++11',
	'-Wno-write-strings', 
	'-DKALDI_DOUBLEPRECISION=0', 
	'-DHAVE_EXECINFO_H=1', 
	'-DHAVE_CXXABI_H', 
	'-DHAVE_ATLAS', 
	'-DKALDI_PARANOID'
]
extra_link_args = []

# Properties for matrix module
matrix_compile_args = []
matrix_libraries = ['kaldi-matrix', 'kaldi-base']
matrix_link_args = []
matrix_sources = [
					'build/kaldi/matrix/kaldi-vector.cc',
					'build/kaldi/matrix/kaldi-vector_init.cc',
					os.path.join(opt, 'clif/python/runtime.cc'),
					os.path.join(opt, 'clif/python/slots.cc'),
					os.path.join(opt, 'clif/python/types.cc'),
				 ]


if DEBUG:
	extra_compile_args += ['-O0', '-g']
	extra_link_args += ['-O0', '-g']
	
################################################################################
# Declare extensions and package
################################################################################
extensions = []
packages = find_packages()
matrix = Extension("matrix",
			  libraries = matrix_sources,
			  sources = matrix_sources,
			  language = 'c++',
			  extra_compile_args = matrix_compile_args + extra_compile_args,
			  include_dirs = include_dirs,
			  library_dirs = library_dirs,
			  extra_link_args = matrix_link_args + extra_link_args)
extensions.append(C)

setup(name = 'pykaldi',
	  version = '0.0.1',
	  description='Kaldi Python Wrapper',
	  author='SAIL',
	  ext_modules=extensions,
	  cmdclass= {
					'build_deps': build_deps,
				},
	  packages=packages,
	  package_data={},
	  install_requires=['enum34;python_version<"3.4"'],
	  )
