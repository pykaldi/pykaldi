import sys
import os

# This is needed for extension libs to be able to load each other.
root = os.path.dirname(__file__)
for entry in os.listdir(root):
    path = os.path.join(root, entry)
    if os.path.isdir(path):
        sys.path.append(path)
