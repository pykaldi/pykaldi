import os
import kaldi
import pkgutil
import argparse 
from subprocess import check_output, check_call
from tempfile import NamedTemporaryFile

parser = argparse.ArgumentParser(description = "Generates autosummay documentation for modules in pykaldi")
# parser.add_argument('--force', '-f', action='store_true', help = 'Overwrite files in _autosummary. Defaults false.')
parser.add_argument('--outputdir', '-o', dest = 'out_dir', default = '_autosummary', help = 'Output directory. Defaults to _autosummary' )
parser.add_argument('--include_private', action='store_true', help = 'Include private modules. Defaults false.')
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

##################################################
# Generate API definition with autogen
##################################################
baseRST = [".. autosummary::",
           "  :toctree: {}".format(args.out_dir),
           ""]
for importer, modname, ispkg in pkgutil.walk_packages(path=kaldi.__path__,
                                                      prefix=kaldi.__name__+'.',
                                                      onerror=lambda x: None):

    if modname.split(".")[-1][0] == "_" and not args.include_private:
        continue

    baseRST.append("  {}".format(modname))

baseRST = "\n".join(baseRST)
print(baseRST)

temp = NamedTemporaryFile(mode='w+t')

try:
    temp.write(baseRST)
    temp.seek(0)

    ##################################################
    # Call autogen on baseRST
    ##################################################
    check_call(['sphinx-autogen', '--output-dir={}'.format(args.out_dir), temp.name])

finally:
    temp.close()


##################################################
# For things listed in API.rst, 
# Include its submodules
##################################################
print("Writing API.rst")
API = [".. autosummary::",
       "  :toctree: {}".format(args.out_dir),
       ""]
for importer, modname, ispkg in pkgutil.walk_packages(path=kaldi.__path__,
                                                      prefix=kaldi.__name__+'.',
                                                      onerror=lambda x: None):
    
    if modname.split(".")[-1][0] == "_" and not args.include_private:
        continue

    if ispkg:
        
        # Add it to the API.rst
        API.append("  {}".format(modname))

    else:
        rst_f = "{}.rst".format(modname)
        rst_path = os.path.join(args.out_dir, rst_f)

        pkg_f = "{}.rst".format(".".join(modname.split(".")[:-1]))
        pkg_path = os.path.join(args.out_dir, pkg_f)

        print("Modifying {}".format(rst_path))
        print("\tFrom {} module".format(pkg_f))

        # Modify file so it has a subheader
        check_call(['sed', '-i', 's/=/-/g', rst_path])

        # Include it in pkg.RST
        with open(pkg_path, "a") as inpt:
            inpt.write("""\n.. include:: {}\n\n""".format(rst_f))

# Save API to file
with open("API.rst", "w") as outpt:
    outpt.write("\n".join(API))

