#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
import pkgutil
import sys

from subprocess import check_call

import kaldi


parser = argparse.ArgumentParser(
    description="Generates autosummary documentation for pykaldi.")
# parser.add_argument('--force', '-f', action='store_true',
#                     help='Overwrite files. Default: False.')
parser.add_argument('--out_dir', '-o', default='api',
                    help='Output directory. Default: api' )
parser.add_argument('--include_private', action='store_true',
                    help='Include private modules. Default: False.')
args = parser.parse_args()

if os.path.exists(args.out_dir):
    print("Output directory: {} already exists.".format(args.out_dir),
          file=sys.stderr)
    sys.exit(1)

os.mkdir(args.out_dir)


##################################################
# Generate autosummary lists and api
##################################################

with open("api.rst", "w") as api, \
     open("packages.rst", "w") as packages, \
     open("modules.rst", "w") as modules:

    print(".. toctree::\n   :caption: API Guide\n   :hidden:\n", file=api)
    # print("   {}/kaldi".format(args.out_dir), file=api)
    print(".. autosummary::\n   :toctree: {}\n".format(args.out_dir),
          file=packages)
    print(".. autosummary::\n   :toctree: {}\n".format(args.out_dir),
          file=modules)

    for _, modname, ispkg in pkgutil.walk_packages(path=kaldi.__path__,
                                                   prefix=kaldi.__name__+'.',
                                                   onerror=lambda x: None):
        if modname.split(".")[-1][0] == "_" and not args.include_private:
            continue
        if modname == "kaldi.itf":
            continue
        if ispkg:
            print("   {}/{}".format(args.out_dir, modname), file=api)
            print("   {}".format(modname), file=packages)
        else:
            if len(modname.split(".")) == 2:
                print("   {}/{}".format(args.out_dir, modname), file=api)
            print("   {}".format(modname), file=modules)

##################################################
# Call autogen
##################################################

check_call(['sphinx-autogen', '-i', '-o', args.out_dir, 'packages.rst'])
check_call(['sphinx-autogen', '-i', '-o', args.out_dir, 'modules.rst'])
check_call(['rm' , '-f', 'packages.rst', 'modules.rst'])

##################################################
# Include submodules in package documentation
##################################################

for importer, modname, ispkg in pkgutil.walk_packages(path=kaldi.__path__,
                                                      prefix=kaldi.__name__+'.',
                                                      onerror=lambda x: None):
    if modname.split(".")[-1][0] == "_" and not args.include_private:
        continue
    if modname == "kaldi.itf":
        continue
    if not ispkg and len(modname.split(".")) > 2:
        mod_file = "{}.rst".format(modname)
        mod_path = os.path.join(args.out_dir, mod_file)

        pkg_file = "{}.rst".format(".".join(modname.split(".")[:-1]))
        pkg_path = os.path.join(args.out_dir, pkg_file)

        # Edit submodule headers
        check_call(['sed', '-i', 's/=/-/g', mod_path])

        # Include submodule in pkg.rst
        with open(pkg_path, "a") as pkg:
            # pkg.write("""\n.. include:: {}\n\n""".format(mod_file))
            pkg.write("\n")
            pkg.write(open(mod_path).read())

        # Remove mod.rst
        check_call(['rm', '-f', mod_path])

##################################################
# Add autosummary nosignatures option
##################################################

for importer, modname, ispkg in pkgutil.walk_packages(path=kaldi.__path__,
                                                      prefix=kaldi.__name__+'.',
                                                      onerror=lambda x: None):
    if modname.split(".")[-1][0] == "_" and not args.include_private:
        continue
    if modname == "kaldi.itf":
        continue
    if ispkg:
        pkg_file = "{}.rst".format(modname)
        pkg_path = os.path.join(args.out_dir, pkg_file)

        check_call(['sed', '-i',
                    's/autosummary::/autosummary::\\n      :nosignatures:/g',
                     pkg_path])
