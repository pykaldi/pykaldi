#!/bin/bash

# 
# 
# Installation script for PyKaldi
# 
# Usage:
# 	./INSTALL.sh [python]
# 
# 	python - Python executable to use. Defaults to current python.
# 
# 

set -e -x 

#######################################################################################################
# Check if we are in the pykaldi directory
# Else error out
#######################################################################################################
# Gets absolute script directory regardless of where it was called from
script_full_path=$(cd $(dirname "$0"); pwd)

if [ ! "$script_full_path" = "$PWD" ]; then
	echo "Change directory to PyKaldi directory before running this script."
	exit 1
fi

#######################################################################################################
# Installation configuration
# These determine where things are going to get installed
#######################################################################################################
PYKALDI_DIR="$PWD"
TOOLS_DIR="$PYKALDI_DIR/extras/tools"
PROTOBUF_DIR="$TOOLS_DIR/protobuf"
CLIFSRC_DIR="$TOOLS_DIR/clif"
KALDI_DIR="$TOOLS_DIR/kaldi"

PYTHON_EXECUTABLE=$(which python)
if [[ -n "$1" ]]; then
	PYTHON_EXECUTABLE="$1"
	shift
fi

####################################################################
# Check dependencies
####################################################################
if ! $TOOLS_DIR/check_dependencies.sh; then
    exit 1
fi

####################################################################
# Dependencies
# Call installers
####################################################################

# Protobuf
# ---------
protobuf_installed=false
if [ -d "$PROTOBUF_DIR" ] && [ -f "$PROTOBUF_DIR/.DONE" ]; then
	protobuf_installed=true
elif [ -d "$PROTOBUF_DIR" ]; then
	rm -rf "$PROTOBUF_DIR"
fi

if ! $protobuf_installed; then
	$TOOLS_DIR/install_protobuf.sh $PROTOBUF_DIR $PYTHON_EXECUTABLE || exit 1
fi

# clif
# -----------------------
clif_installed=false
if [ -d "$CLIFSRC_DIR" ] && [ -f "$CLIFSRC_DIR/.DONE" ]; then
	clif_installed=true
elif [ -d "$CLIFSRC_DIR" ]; then
	rm -rf "$CLIFSRC_DIR"
	if [ -d "$CLIFSRC_DIR/../clif_backend" ]; then
		rm -rf "$CLIFSRC_DIR/../clif_backend"
	fi
fi

if ! $clif_installed; then
	$TOOLS_DIR/install_clif.sh $CLIFSRC_DIR || exit 1
fi

# Kaldi
# -----------------------
kaldi_installed=false
if [ -d "$KALDI_DIR" ] && [ -f "$KALDI_DIR/.DONE" ]; then
	kaldi_installed=true
elif [ -d "$KALDI_DIR" ]; then
	rm -rf "$KALDI_DIR"
fi

if ! $kaldi_installed; then
	$TOOLS_DIR/install_kaldi.sh $KALDI_DIR || exit 1
fi

####################################################################
# Set env variables
####################################################################
export PATH="$PATH:$CLIF_DIR/clif/bin"
export DEBUG=0

###########################################################################
# If you ever get to this point and you have not downloaded pykaldi repo yet:
# 1) How? Why?...
# 2) Just uncomment the next two lines...
###########################################################################
# git clone $PYKALDI_GIT $PYKALDI_DIR
# cd $PYKALDI_DIR
############################################################################

# Install pykaldi
python setup.py install 

cat <<EOF

Done installing PyKaldi!

Remember to update your \$PATH:

export PATH="\$PATH:$CLIF_DIR/clif/bin"


EOF
exit 0
