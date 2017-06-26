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

DEBUG = check_env_flag('DEBUG')
PYCLIF_DIR = os.getenv('PYCLIF')
KALDI_DIR = os.getenv('KALDI')

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
		build_all_cmd = ['bash', 'build_all.sh', PYCLIF_DIR, KALDI_DIR]
		if subprocess.call(build_all_cmd) != 0:
			sys.exit(1)

class build_module(Command):
	user_options = []

	def initialize_options(self):
		pass

	def finalize_options(self):
		pass

	def run(self):
		self.run_command('build_py')
		self.run_command('build_ext')

class develop(setuptools.command.develop.develop):

	def run(self):
		build_py.create_version_file()
		setuptools.command.develop.develop.run(self)

class build_ext(setuptools.command.build_ext.build_ext):

	def run(self):
		# It's an old-style class in Python 2.7...
		setuptools.command.build_ext.build_ext.run(self)


class build(distutils.command.build.build):
	sub_commands = [
		('build_deps', lambda self: True),
	] + distutils.command.build.build.sub_commands


class install(setuptools.command.install.install):

	def run(self):
		if not self.skip_build:
			self.run_command('build_deps')
		setuptools.command.install.install.run(self)


class clean(distutils.command.clean.clean):

	def run(self):
		import glob
		with open('.gitignore', 'r') as f:
			ignores = f.read()
			for wildcard in filter(bool, ignores.split('\n')):
				for filename in glob.glob(wildcard):
					try:
						os.remove(filename)
					except OSError:
						shutil.rmtree(filename, ignore_errors=True)

		# It's an old-style class in Python 2.7...
		distutils.command.clean.clean.run(self)

################################################################################
# Configure compile flags
################################################################################
library_dirs = ['/saildisk/tools/kaldi/src/lib/']


cwd = os.path.dirname(os.path.abspath(__file__))

include_dirs = [
	cwd,
	# Path to clif runtime headers and example cc lib headers
	user_home + '/opt',
	'/saildisk/tools/kaldi/src/',
	'/saildisk/tools/kaldi/tools/openfst/include',
	'/saildisk/tools/kaldi/tools/ATLAS/include'
]

main_compile_args = []
main_libraries = ['kaldi-matrix', 'kaldi-base']
main_link_args = []
main_sources = [
	'build/kaldi/matrix/kaldi-vector.cc',
	'build/kaldi/matrix/kaldi-vector_init.cc',
	user_home + '/opt/clif/python/runtime.cc',
	user_home + '/opt/clif/python/slots.cc',
	user_home + '/opt/clif/python/types.cc',
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

if DEBUG:
    extra_compile_args += ['-O0', '-g']
    extra_link_args += ['-O0', '-g']

################################################################################
# Declare extensions and package
################################################################################
extensions = []
packages = find_packages()
C = Extension("pykaldi._C",
			  libraries = main_libraries,
			  sources = main_sources,
			  language = 'c++',
			  extra_compile_args = extra_compile_args,
			  include_dirs = include_dirs,
			  library_dirs = library_dirs,
			  extra_link_args = extra_link_args)
extensions.append(C)

setup(name = 'pykaldi',
	  version = '0.0.1',
	  description='Kaldi Python Wrapper',
	  author='SAIL',
	  ext_modules=extensions,
	  cmdclass= {
	  				'build': build,
	  				'build_py': build_py,
	  				'build_ext': build_ext,
	  				'build_deps': build_deps,
	  				'build_module': build_module,
	  				'develop': develop,
	  				'install': install,
	  				'clean': clean
		  		},
	  packages=packages,
	  package_data={},
	  install_requires=['enum34;python_version<"3.4"'],
	  )