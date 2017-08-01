#!/usr/bin/env python
"""Setup configuration."""

from setuptools import setup, find_packages
import setuptools.command.build_ext
import setuptools.command.install_lib
import distutils.command.build
import setuptools.extension

import subprocess #Needed for line 78
from subprocess import check_output
import os

import numpy

################################################################################
# Check variables / find programs
################################################################################

DEBUG = os.getenv('DEBUG') in ['ON', '1', 'YES', 'TRUE', 'Y']
PYCLIF = os.getenv("PYCLIF")
CLIF_DIR = os.getenv('CLIF_DIR')
KALDI_DIR = os.getenv('KALDI_DIR')
CWD = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(CWD, 'build')

if not PYCLIF:
    try:
        PYCLIF = check_output(['which', 'pyclif']).decode("utf-8").strip()
    except OSError:
        raise RuntimeError("Could not find pyclif. Please add pyclif binary to "
                           "your PATH or set PYCLIF environment variable.")

if not KALDI_DIR:
  raise RuntimeError("KALDI_DIR environment variable is not set.")

if not CLIF_DIR:
    CLIF_DIR = os.path.dirname(os.path.dirname(PYCLIF))
    print("CLIF_DIR environment variable is not set.")
    print("Defaulting to {}".format(CLIF_DIR))

NUMPY_INC_DIR = numpy.get_include()

if DEBUG:
    print("#"*50)
    print("CWD: {}".format(CWD))
    print("PYCLIF: {}".format(PYCLIF))
    print("KALDI_DIR: {}".format(KALDI_DIR))
    print("CLIF_DIR: {}".format(CLIF_DIR))
    print("NUMPY_INC_DIR: {}".format(NUMPY_INC_DIR))
    print("CLIF_CXX_FLAGS: {}".format(os.getenv("CLIF_CXX_FLAGS")))
    print("BUILD_DIR: {}".format(BUILD_DIR))
    print("#"*50)

################################################################################
# Use CMake to build Python extensions in parallel
################################################################################

class KaldiExtension(setuptools.extension.Extension):
    """Dummy class that only holds the name of the extension"""
    def __init__(self, name):
        setuptools.extension.Extension.__init__(self, name, [])

class CMakeBuild(setuptools.command.build_ext.build_ext):
    def run(self):
        old_inplace, self.inplace = self.inplace, 0

        cmake_args = ['-DKALDI_DIR=' + KALDI_DIR,
                      '-DPYCLIF=' + PYCLIF,
                      '-DCLIF_DIR=' + CLIF_DIR,
                      '-DCLIF_CXX_FLAGS=' + os.getenv("CLIF_CXX_FLAGS", ""), #CLIF_CXX_FLAGS might not be set
                      '-DNUMPY_INC_DIR='+NUMPY_INC_DIR]
        if DEBUG:
            cmake_args += ['-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON']

        if not os.path.exists(BUILD_DIR):
            os.makedirs(BUILD_DIR)

        subprocess.check_call(['cmake', '..'] + cmake_args, cwd = BUILD_DIR)
        subprocess.check_call(['make', '-j'], cwd = BUILD_DIR)
        print() # Add an empty line for cleaner output

        self.inplace = old_inplace
        if old_inplace:
            self.copy_extensions_to_source()

    def get_ext_filename(self, fullname):
        """Convert the name of an extension (eg. "foo.bar") into the name
        of the file from which it will be loaded (eg. "foo/bar.so"). This
        patch overrides platform specific extension suffix with ".so".
        """
        ext_path = fullname.split('.')
        ext_suffix = '.so'
        return os.path.join(*ext_path) + ext_suffix

class build(distutils.command.build.build):
    def finalize_options(self):
        self.build_base = 'build'
        self.build_lib = 'build/lib'
        distutils.command.build.build.finalize_options(self)

class install_lib(setuptools.command.install_lib.install_lib):
    def install(self):
        self.build_dir = 'build/lib'
        outfiles = setuptools.command.install_lib.install_lib.install(self)
        print(outfiles)

################################################################################
# Setup pykaldi
################################################################################

extensions = [
                KaldiExtension("kaldi._clif"),
                KaldiExtension("kaldi.fstext.symbol_table"),
                KaldiExtension("kaldi.fstext.weight"),
                KaldiExtension("kaldi.fstext.float_weight"),
                KaldiExtension("kaldi.fstext.lattice_weight"),
                KaldiExtension("kaldi.fstext.lattice_utils"),
                KaldiExtension("kaldi.fstext.fst_types"),
                KaldiExtension("kaldi.fstext.fst_ext"),
                KaldiExtension("kaldi.fstext.kaldi_fst_io"),
                KaldiExtension("kaldi.matrix.matrix_common"),
                KaldiExtension("kaldi.matrix.kaldi_vector"),
                KaldiExtension("kaldi.matrix.kaldi_matrix"),
                KaldiExtension("kaldi.matrix.matrix_ext"),
                KaldiExtension("kaldi.matrix.compressed_matrix"),
                KaldiExtension("kaldi.matrix.packed_matrix"),
                KaldiExtension("kaldi.matrix.sp_matrix"),
                KaldiExtension("kaldi.matrix.tp_matrix"),
                KaldiExtension("kaldi.matrix.kaldi_vector_ext"),
                KaldiExtension("kaldi.matrix.matrix_functions"),
                KaldiExtension("kaldi.feat.feature_window"),
                KaldiExtension("kaldi.feat.feature_functions"),
                KaldiExtension("kaldi.feat.mel_computations"),
                KaldiExtension("kaldi.feat.feature_mfcc"),
                KaldiExtension("kaldi.feat.feature_common_ext"),
                KaldiExtension("kaldi.feat.wave_reader"),
                KaldiExtension("kaldi.util.kaldi_io"),
                KaldiExtension("kaldi.util.kaldi_holder"),
                KaldiExtension("kaldi.util.kaldi_table"),
                KaldiExtension("kaldi.util.kaldi_table_ext"),
             ]

packages = find_packages()

setup(name = 'pykaldi',
      version = '0.0.2',
      description = 'Kaldi Python Wrapper',
      author = 'SAIL',
      ext_modules=extensions,
      cmdclass = {
          'build_ext': CMakeBuild,
          'build': build,
          'install_lib': install_lib
          },
      packages = packages,
      package_data = {},
      install_requires = ['enum34;python_version<"3.4"', 'numpy'])
