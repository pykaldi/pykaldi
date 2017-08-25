import sys
import os

__version__ = '0.0.2'

# This is needed for extension libs to be able to load each other.
root = os.path.dirname(__file__)
for entry in os.listdir(root):
    path = os.path.join(root, entry)
    if os.path.isdir(path):
        sys.path.append(path)
