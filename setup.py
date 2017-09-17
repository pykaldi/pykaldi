#!/usr/bin/env python
"""Setup configuration."""
from __future__ import print_function

import os

import distutils.command.build
import setuptools.command.build_ext
import setuptools.command.install_lib
import setuptools.extension

from distutils.file_util import copy_file
from setuptools import setup, find_packages, Command
from subprocess import check_output, check_call

import numpy

################################################################################
# Check variables / find programs
################################################################################

DEBUG = os.getenv('DEBUG').upper() in ['ON', '1', 'YES', 'TRUE', 'Y']
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
    CUDA = check_output(['make', 'print-CUDA']).decode("utf-8").strip()
    KALDI_HAVE_CUDA = CUDA.upper() in ['ON', '1', 'YES', 'TRUE', 'Y']
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

class Extension(setuptools.extension.Extension):
    """Dummy extension class that only holds the name of the extension."""
    def __init__(self, name):
        setuptools.extension.Extension.__init__(self, name, [])
        self._needs_stub = False
    def __str__(self):
        return "Extension({})".format(self.name)


def populateExtensionList():
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

        # Populates extension list
        self.extensions = populateExtensionList()

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
            check_call(['make', 'docs'], cwd = BUILD_DIR)
        except ImportError:
            print("Sphinx was not found. Install it using pip install sphinx.")


class install_lib(setuptools.command.install_lib.install_lib):
    def install(self):
        self.build_dir = 'build/lib'
        outfiles = setuptools.command.install_lib.install_lib.install(self)
        print(outfiles)

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
      description = 'Kaldi Python Wrapper',
      author = 'SAIL',
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
