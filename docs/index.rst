.. PyKaldi documentation master file.

:github_url: https://github.com/pykaldi/pykaldi/blob/master/docs/index.rst

PyKaldi Documentation
---------------------

`PyKaldi <https://github.com/pykaldi/pykaldi>`_ is a Python wrapper for
`Kaldi <http://kaldi-asr.org>`_. It aims to bridge the gap between Kaldi and
all the nice things Python has to offer. Its main features are:

* Near-complete coverage of Kaldi C++ API

* First class support for Kaldi and OpenFst types in Python

* Extensible design

* Open license

* Extensive documentation

* Thorough testing

* Example scripts

* Support for both Python 2.7 and 3.5+


.. seealso::

   `Kaldi Documentation <http://kaldi-asr.org/doc/index.html>`__
     PyKaldi API matches Kaldi API to a large extent, hence most of Kaldi
     documentation applies to PyKaldi verbatim. Further, Kaldi documentation
     includes detailed descriptions of the library API, the algorithms used and
     the software architecture, which are currently significantly more
     comprehensive than what PyKaldi documentation provides.


.. toctree::
   :caption: User Guide
   :glob:
   :hidden:
   :maxdepth: 2

   user/*

.. toctree::
   :caption: Developer Guide
   :glob:
   :hidden:
   :maxdepth: 2

   dev/*

.. include:: api.rst
