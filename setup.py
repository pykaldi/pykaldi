#!/usr/bin/env python
"""Setup configuration."""
from __future__ import print_function

from setuptools import setup, find_packages
import setuptools.command.build_ext
import setuptools.command.install_lib
import distutils.command.build
import setuptools.extension

import os
from subprocess import check_output, check_call

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
CLIF_CXX_FLAGS = os.getenv("CLIF_CXX_FLAGS", "")

if not PYCLIF:
    try:
        PYCLIF = check_output(['which', 'pyclif']).decode("utf-8").strip()
    except OSError:
        raise RuntimeError("Could not find pyclif. Please add pyclif binary to"
                           " your PATH or set PYCLIF environment variable.")

if KALDI_DIR:
    KALDI_MK_PATH = os.path.join(KALDI_DIR, "src", "kaldi.mk")
    with open("Makefile", "w") as makefile:
        print("include {}".format(KALDI_MK_PATH), file=makefile)
        print("print-% : ; @echo $($*)", file=makefile)
    CXX_FLAGS = check_output(['make', 'print-CXXFLAGS']).decode("utf-8").strip()
    KALDI_HAVE_CUDA = int(check_output(['make', 'print-CUDA']).decode("utf-8").strip())
    check_call(["rm", "Makefile"])
else:
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
    print("CXX_FLAGS: {}".format(CXX_FLAGS))
    print("CLIF_CXX_FLAGS: {}".format(CLIF_CXX_FLAGS))
    print("BUILD_DIR: {}".format(BUILD_DIR))
    print("KALDI_HAVE_CUDA: {}".format(KALDI_HAVE_CUDA))
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
                      '-DCXX_FLAGS=' + CXX_FLAGS,
                      '-DCLIF_CXX_FLAGS=' + CLIF_CXX_FLAGS,
                      '-DNUMPY_INC_DIR='+ NUMPY_INC_DIR,
                      '-DCUDA=TRUE' if KALDI_HAVE_CUDA else '-DCUDA=FALSE']
        if DEBUG:
            cmake_args += ['-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON']

        if not os.path.exists(BUILD_DIR):
            os.makedirs(BUILD_DIR)

        check_call(['cmake', '..'] + cmake_args, cwd = BUILD_DIR)
        check_call(['make', '-j'], cwd = BUILD_DIR)
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
                KaldiExtension("kaldi.base.timer"),
                KaldiExtension("kaldi.base.io_funcs"),
                KaldiExtension("kaldi.base.kaldi_math"),
                KaldiExtension("kaldi.itf.options_itf"),
                KaldiExtension("kaldi.itf.context_dep_itf"),
                KaldiExtension("kaldi.itf.decodable_itf"),
                KaldiExtension("kaldi.itf.online_feature_itf"),
                KaldiExtension("kaldi.fstext.properties"),
                KaldiExtension("kaldi.fstext.symbol_table"),
                KaldiExtension("kaldi.fstext.weight"),
                KaldiExtension("kaldi.fstext.float_weight"),
                KaldiExtension("kaldi.fstext.lattice_weight"),
                KaldiExtension("kaldi.fstext.lattice_utils"),
                KaldiExtension("kaldi.fstext.fst"),
                KaldiExtension("kaldi.fstext.expanded_fst"),
                KaldiExtension("kaldi.fstext.mutable_fst"),
                KaldiExtension("kaldi.fstext.vector_fst"),
                KaldiExtension("kaldi.fstext.fst_ext"),
                KaldiExtension("kaldi.fstext.kaldi_fst_io"),
                KaldiExtension("kaldi.fstext.fstext_utils"),
                KaldiExtension("kaldi.fstext.drawer"),
                KaldiExtension("kaldi.fstext.printer"),
                KaldiExtension("kaldi.fstext.compiler"),
                KaldiExtension("kaldi.fstext.getters"),
                KaldiExtension("kaldi.fstext.encode"),
                KaldiExtension("kaldi.fstext.fst_operations"),
                KaldiExtension("kaldi.matrix.matrix_common"),
                KaldiExtension("kaldi.matrix.kaldi_vector"),
                KaldiExtension("kaldi.matrix.kaldi_matrix"),
                KaldiExtension("kaldi.matrix.matrix_ext"),
                KaldiExtension("kaldi.matrix.compressed_matrix"),
                KaldiExtension("kaldi.matrix.packed_matrix"),
                KaldiExtension("kaldi.matrix.sp_matrix"),
                KaldiExtension("kaldi.matrix.tp_matrix"),
                KaldiExtension("kaldi.matrix.kaldi_vector_ext"),
                KaldiExtension("kaldi.matrix.kaldi_matrix_ext"),
                KaldiExtension("kaldi.matrix.matrix_functions"),
                KaldiExtension("kaldi.feat.resample"),
                KaldiExtension("kaldi.feat.signal"),
                KaldiExtension("kaldi.feat.feature_window"),
                KaldiExtension("kaldi.feat.feature_functions"),
                KaldiExtension("kaldi.feat.mel_computations"),
                KaldiExtension("kaldi.feat.feature_spectrogram"),
                KaldiExtension("kaldi.feat.feature_mfcc"),
                KaldiExtension("kaldi.feat.feature_plp"),
                KaldiExtension("kaldi.feat.feature_fbank"),
                KaldiExtension("kaldi.feat.feature_common_ext"),
                KaldiExtension("kaldi.feat.online_feature"),
                KaldiExtension("kaldi.feat.pitch_functions"),
                KaldiExtension("kaldi.feat.wave_reader"),
                KaldiExtension("kaldi.util.options_ext"),
                KaldiExtension("kaldi.util.kaldi_io"),
                KaldiExtension("kaldi.util.kaldi_holder"),
                KaldiExtension("kaldi.util.kaldi_table"),
                KaldiExtension("kaldi.util.iostream"),
                KaldiExtension("kaldi.util.fstream"),
                KaldiExtension("kaldi.util.sstream"),
                KaldiExtension("kaldi.gmm.model_common"),
                KaldiExtension("kaldi.gmm.diag_gmm"),
                KaldiExtension("kaldi.gmm.full_gmm"),
                KaldiExtension("kaldi.gmm.full_gmm_ext"),
                KaldiExtension("kaldi.gmm.full_gmm_normal"),
                KaldiExtension("kaldi.gmm.mle_diag_gmm"),
                KaldiExtension("kaldi.gmm.am_diag_gmm"),
                KaldiExtension("kaldi.gmm.decodable_am_diag_gmm"),
                KaldiExtension("kaldi.gmm.mle_full_gmm"),
                KaldiExtension("kaldi.gmm.tests.model_test_common"),
                KaldiExtension("kaldi.hmm.hmm_topology"),
                KaldiExtension("kaldi.hmm.transition_model"),
                KaldiExtension("kaldi.decoder.faster_decoder"),
                KaldiExtension("kaldi.cudamatrix.cu_matrixdim"),
                KaldiExtension("kaldi.cudamatrix.cu_array"),
                KaldiExtension("kaldi.cudamatrix.cu_vector"),
                KaldiExtension("kaldi.cudamatrix.cu_matrix")
             ]

if KALDI_HAVE_CUDA:
    extensions.append(KaldiExtension("kaldi.cudamatrix.cu_device"))


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
      install_requires = ['enum34;python_version<"3.4"', 'numpy'],
      zip_safe = False)
