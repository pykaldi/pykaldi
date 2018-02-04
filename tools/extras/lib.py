import distutils.sysconfig as du_sysconfig
import itertools
import os
import sys
import sysconfig

def get_python_version():
    """Get version associated with the current python interpreter."""
    python_version = sysconfig.get_config_var('VERSION')

    if not python_version:
        python_version = sysconfig.get_config_var('py_version_short')

    if not python_version:
        python_version = ".".join(map(str, sys.version_info[:2]))

    return python_version

def get_python_library(python_version):
    """Get path to the python library associated with the current python
    interpreter."""
    # determine direct path to libpython
    python_library = sysconfig.get_config_var('LIBRARY')

    # if static (or nonexistent), try to find a suitable dynamic libpython
    if (python_library is None or
            os.path.splitext(python_library)[1][-2:] == '.a'):

        candidate_lib_prefixes = ['', 'lib']

        candidate_extensions = ['.lib', '.so', '.a']
        if sysconfig.get_config_var('WITH_DYLD'):
            candidate_extensions.insert(0, '.dylib')

        candidate_versions = [python_version]
        if python_version:
            candidate_versions.append('')
            candidate_versions.insert(
                0, "".join(python_version.split(".")[:2]))

        abiflags = getattr(sys, 'abiflags', '')
        candidate_abiflags = [abiflags]
        if abiflags:
            candidate_abiflags.append('')

        # Ensure the value injected by virtualenv is
        # returned on windows.
        # Because calling `sysconfig.get_config_var('multiarchsubdir')`
        # returns an empty string on Linux, `du_sysconfig` is only used to
        # get the value of `LIBDIR`.
        libdir = du_sysconfig.get_config_var('LIBDIR')
        if sysconfig.get_config_var('MULTIARCH'):
            masd = sysconfig.get_config_var('multiarchsubdir')
            if masd:
                if masd.startswith(os.sep):
                    masd = masd[len(os.sep):]
                libdir = os.path.join(libdir, masd)

        if libdir is None:
            libdir = os.path.abspath(os.path.join(
                sysconfig.get_config_var('LIBDEST'), "..", "libs"))

        candidates = (
            os.path.join(
                libdir,
                ''.join((pre, 'python', ver, abi, ext))
            )
            for (pre, ext, ver, abi) in itertools.product(
                candidate_lib_prefixes,
                candidate_extensions,
                candidate_versions,
                candidate_abiflags
            )
        )

        for candidate in candidates:
            if os.path.exists(candidate):
                # we found a (likely alternate) libpython
                python_library = candidate
                break

    # TODO(opadron): what happens if we don't find a libpython?

    return python_library

python_version = get_python_version()
python_library = get_python_library(python_version)

print(python_library)
