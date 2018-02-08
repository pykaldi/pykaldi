#!/usr/bin/env python
"""Setup configuration."""
from __future__ import print_function

import os
import sys

import distutils.command.build
import setuptools.command.build_ext
import setuptools.command.install_lib
import setuptools.extension

from distutils.file_util import copy_file
from setuptools import setup, find_packages, Command
from subprocess import check_output, check_call, CalledProcessError

################################################################################
# Check variables / find programs
################################################################################

DEBUG = os.getenv('DEBUG', 'NO').upper() in ['ON', '1', 'YES', 'TRUE', 'Y']
PYCLIF = os.getenv("PYCLIF")
CLIF_MATCHER = os.getenv('CLIF_MATCHER')
KALDI_DIR = os.getenv('KALDI_DIR')
CWD = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(CWD, 'build')
NPROC = check_output(['getconf', '_NPROCESSORS_ONLN']).decode("utf-8").strip()
MAKE_NUM_JOBS = os.getenv('MAKE_NUM_JOBS', NPROC)


if not PYCLIF:
    PYCLIF = os.path.join(sys.prefix, 'bin/clif-matcher')

if not (os.path.isfile(PYCLIF) and os.access(PYCLIF, os.X_OK)):
    try:
        PYCLIF = check_output(['which', 'pyclif']).decode("utf-8").strip()
    except OSError:
        raise RuntimeError("Could not find pyclif. Please add pyclif binary to"
                           " your PATH or set PYCLIF environment variable.")

if not CLIF_MATCHER:
    CLIF_MATCHER = os.path.join(sys.prefix, 'clang/bin/clif-matcher')

if not (os.path.isfile(CLIF_MATCHER) and os.access(CLIF_MATCHER, os.X_OK)):
    raise RuntimeError("Could not find clif-matcher. Please make sure CLIF "
                       "was installed under the current python environment "
                       "or set CLIF_MATCHER environment variable.")

CLANG = os.path.join(os.path.dirname(CLIF_MATCHER), "clang")
RESOURCE_DIR = check_output("echo '#include <limits.h>' | {} -xc -v - 2>&1 "
                            "| tr ' ' '\n' | grep -A1 resource-dir | tail -1"
                            .format(CLANG), shell=True).decode("utf-8").strip()
CLIF_CXX_FLAGS="-I{}/include".format(RESOURCE_DIR)

if not KALDI_DIR:
    KALDI_DIR = os.path.join(CWD, "tools/kaldi")

KALDI_MK_PATH = os.path.join(KALDI_DIR, "src", "kaldi.mk")
if not os.path.isfile(KALDI_MK_PATH):
  raise RuntimeError("Could not find Kaldi installation. Please install Kaldi "
                     "under the tools directory or set KALDI_DIR environment "
                     "variable to where Kaldi is installed.")

with open("Makefile", "w") as makefile:
    print("include {}".format(KALDI_MK_PATH), file=makefile)
    print("print-% : ; @echo $($*)", file=makefile)
CXX_FLAGS = check_output(['make', 'print-CXXFLAGS']).decode("utf-8").strip()
CUDA = check_output(['make', 'print-CUDA']).decode("utf-8").strip()
KALDI_CUDA = CUDA.upper() in ['ON', '1', 'YES', 'TRUE', 'Y']
check_call(["rm", "Makefile"])

TFRNNLM_LIB_PATH = os.path.join(KALDI_DIR, "src", "lib",
                                "libkaldi-tensorflow-rnnlm.so")
KALDI_TFRNNLM = True if os.path.exists(TFRNNLM_LIB_PATH) else False
if KALDI_TFRNNLM:
    with open("Makefile", "w") as makefile:
        TF_DIR = os.path.join(KALDI_DIR, "tools", "tensorflow")
        print("TENSORFLOW = {}".format(TF_DIR), file=makefile)
        TFRNNLM_MK_PATH = os.path.join(KALDI_DIR, "src", "tfrnnlm",
                                       "Makefile")
        for line in open(TFRNNLM_MK_PATH):
            if line.startswith("include") or line.startswith("TENSORFLOW"):
                continue
            print(line, file=makefile, end='')
        print("print-% : ; @echo $($*)", file=makefile)
    TFRNNLM_CXX_FLAGS = check_output(['make', 'print-EXTRA_CXXFLAGS'])
    TFRNNLM_CXX_FLAGS = TFRNNLM_CXX_FLAGS.decode("utf-8").strip()
    TF_LIB_DIR = os.path.join(KALDI_DIR, "tools", "tensorflow",
                              "bazel-bin", "tensorflow")
    check_call(["rm", "Makefile"])

MAKE_ARGS = []
try:
    import ninja
    CMAKE_GENERATOR = '-GNinja'
    MAKE = 'ninja'
    if DEBUG:
        MAKE_ARGS += ['-v']
except ImportError:
    CMAKE_GENERATOR = ''
    MAKE = 'make'
    if MAKE_NUM_JOBS:
        MAKE_ARGS += ['-j', MAKE_NUM_JOBS]

if DEBUG:
    print("#"*50)
    print("CWD:", CWD)
    print("PYCLIF:", PYCLIF)
    print("CLIF_MATCHER:", CLIF_MATCHER)
    print("KALDI_DIR:", KALDI_DIR)
    print("CXX_FLAGS:", CXX_FLAGS)
    print("CLIF_CXX_FLAGS:", CLIF_CXX_FLAGS)
    print("BUILD_DIR:", BUILD_DIR)
    print("KALDI_CUDA:", KALDI_CUDA)
    print("MAKE:", MAKE, *MAKE_ARGS)
    print("#"*50)

################################################################################
# Use CMake to build Python extensions in parallel
################################################################################

class Extension(setuptools.extension.Extension):
    """Dummy extension class that only holds the name of the extension."""
    def __init__(self, name):
        setuptools.extension.Extension.__init__(self, name, [])
        self._needs_stub = False
    def __str__(self):
        return "Extension({})".format(self.name)


def populate_extension_list():
    extensions = []
    lib_dir = os.path.join(BUILD_DIR, "lib")
    for dirpath, _, filenames in os.walk(os.path.join(lib_dir, "kaldi")):

        lib_path = os.path.relpath(dirpath, lib_dir)

        if lib_path == ".":
            lib_path = "kaldi"

        # append filenames
        for f in filenames:
            f = os.path.splitext(f)[0] # remove extension
            ext_name = "{}.{}".format(lib_path, f)
            extensions.append(Extension(ext_name))
    return extensions


class build(distutils.command.build.build):
    def finalize_options(self):
        self.build_base = 'build'
        self.build_lib = 'build/lib'
        distutils.command.build.build.finalize_options(self)


class build_ext(setuptools.command.build_ext.build_ext):
    def run(self):
        old_inplace, self.inplace = self.inplace, 0

        import numpy as np
        CMAKE_ARGS = ['-DKALDI_DIR=' + KALDI_DIR,
                      '-DPYCLIF=' + PYCLIF,
                      '-DCLIF_MATCHER=' + CLIF_MATCHER,
                      '-DCXX_FLAGS=' + CXX_FLAGS,
                      '-DCLIF_CXX_FLAGS=' + CLIF_CXX_FLAGS,
                      '-DNUMPY_INC_DIR='+ np.get_include(),
                      '-DCUDA=TRUE' if KALDI_CUDA else '-DCUDA=FALSE',
                      '-DTFRNNLM=TRUE' if KALDI_TFRNNLM else '-DTFRNNLM=FALSE',
                      '-DDEBUG=TRUE' if DEBUG else '-DDEBUG=FALSE']

        if KALDI_TFRNNLM:
            CMAKE_ARGS +=['-DTFRNNLM_CXX_FLAGS=' + TFRNNLM_CXX_FLAGS,
                          '-DTF_LIB_DIR=' + TF_LIB_DIR]

        if CMAKE_GENERATOR:
            CMAKE_ARGS += [CMAKE_GENERATOR]

        if DEBUG:
            CMAKE_ARGS += ['-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON']

        if not os.path.exists(BUILD_DIR):
            os.makedirs(BUILD_DIR)

        try:
            check_call(['cmake', '..'] + CMAKE_ARGS, cwd = BUILD_DIR)
            check_call([MAKE] + MAKE_ARGS, cwd = BUILD_DIR)
        except CalledProcessError as err:
            # We catch this exception to disable stack trace.
            print(str(err), file=sys.stderr)
            sys.exit(1)
        print() # Add an empty line for cleaner output

        # Populates extension list
        self.extensions = populate_extension_list()

        if DEBUG:
            for ext in self.extensions:
                print(ext)
            self.verbose = True

        self.inplace = old_inplace
        if old_inplace:
            self.copy_extensions_to_source()

    def copy_extensions_to_source(self):
        build_py = self.get_finalized_command('build_py')
        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)
            modpath = fullname.split('.')
            package = '.'.join(modpath[:-1])
            package_dir = build_py.get_package_dir(package)
            dest_filename = os.path.join(package_dir,
                                         os.path.basename(filename))
            src_filename = os.path.join(self.build_lib, filename)

            # Always copy, even if source is older than destination, to ensure
            # that the right extensions for the current Python/platform are
            # used.
            copy_file(
                src_filename, dest_filename, verbose=self.verbose,
                dry_run=self.dry_run
            )
            if ext._needs_stub:
                self.write_stub(package_dir or os.curdir, ext, True)

    def get_ext_filename(self, fullname):
        """Convert the name of an extension (eg. "foo.bar") into the name
        of the file from which it will be loaded (eg. "foo/bar.so"). This
        patch overrides platform specific extension suffix with ".so".
        """
        ext_path = fullname.split('.')
        ext_suffix = '.so'
        return os.path.join(*ext_path) + ext_suffix


class build_sphinx(Command):
    user_options = []
    description = "Builds documentation using sphinx."

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            import sphinx
            check_call([MAKE, 'docs'], cwd = BUILD_DIR)
        except ImportError:
            print("Sphinx was not found. Install it using pip install sphinx.")


class install_lib(setuptools.command.install_lib.install_lib):
    def install(self):
        self.build_dir = 'build/lib'
        setuptools.command.install_lib.install_lib.install(self)

################################################################################
# Setup pykaldi
################################################################################

# We add a 'dummy' extension so that setuptools runs the build_ext step.
extensions = [Extension("kaldi")]

packages = find_packages()

with open(os.path.join('kaldi', '__version__.py')) as f:
    exec(f.read())

setup(name = 'pykaldi',
      version = __version__,
      description = 'A Python wrapper for Kaldi',
      author = 'Dogan Can, Victor Martinez',
      ext_modules=extensions,
      cmdclass = {
          'build': build,
          'build_ext': build_ext,
          'build_sphinx': build_sphinx,
          'install_lib': install_lib,
          },
      packages = packages,
      package_data = {},
      install_requires = ['enum34;python_version<"3.4"', 'numpy'],
      zip_safe = False)
