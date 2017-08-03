# Extending CLIF with C++ libraries

CLIF has a number of C++ types it already
knows,but it can't cover them all. Normally CLIF used for wrapping new types by
creating a .clif file and running CLIF. However when C++ API use a type that
can't be wrapped normally, user has an ability to teach CLIF how to handle such
type by writing a C++ library. This doc explains how to do that.

When CLIF pass data to/from C++ across the language boundary, it does the
conversion between C++ internal storage representation and Python internal
storage representation.
It might be as simple as copying a C++ pointer or as complex as serializing and
deserializing a protocol buffer. So by writing a C++ library the author can
control how CLIF does that but needs to understand the internal
representation of both.

Teaching CLIF a new type includes 3 steps:

  1. Conversion function to pass data from Python to C++ (`Clif_PyObjAs`).
  1. Conversion function to pass data from C++ to Python (`Clif_PyObjFrom`).
  1. Creating a CLIF name to refer to the C++ type from .clif files.

CLIF uses [ADL](http://en.cppreference.com/w/cpp/language/adl) to find
the conversion function, so those functions should be placed in the namespace
where the C++ type we're teaching CLIF about is located.

Below we describe the conversions possible. None of them are required.
Only write what is possible, CLIF might use the presence or absence of certain
conversions as an indicator of the type capabilities.

## Passing data from Python to C++

This kind represented by name `Clif_PyObjAs` and return `bool` to indicate if an
input Python object was successfully passed to C++. When returning `false` the
routine **must** set a Python exception (eg. with
*PyErr_SetString(PyExc_ValueError, "invalid value for CppType")* call).

```c++
bool Clif_PyObjAs(PyObject*, CppType*);
```

Write that function when **copying** data from Python to C++ is possible and
needed. Use `gtl::optional<CppType>*` if CppType is not default constructible.

```c++
bool Clif_PyObjAs(PyObject*, std::unique_ptr<CppType>*);
```

Write that function when **moving** data from Python to C++ is possible and
needed. After the move the Python object will be invalidated and raise an
exception on access attempt.

```c++
bool Clif_PyObjAs(PyObject*, std::shared_ptr<CppType>*);
```

Write that function when **sharing** data ownership between Python and C++ is
possible and needed. The data will be kept alive as long as either side needs
it.

```c++
bool Clif_PyObjAs(PyObject*, CppType**);
```

Write that function when temporary access to Python data from C++ is possible
and needed. This pointer represents *borrowing* data from Python, not an
ownership transfer. This form is giving C++ a raw pointer to the internal
representation of the Python object internals. Be careful as the Python object
can disappear making the pointer invalid.

## Passing data from C++ to Python

This kind represented by name `Clif_PyObjFrom` and return a **new** `PyObject`.
If conversion fails, set Python exception and return `nullptr`.

```c++
PyObject* Clif_PyObjFrom(const CppType&, py::PostProc);
```

Write that function when **copying** data from C++ to Python is possible and
needed.

```c++
PyObject* Clif_PyObjFrom(std::unique_ptr<CppType>, py::PostProc);
```
Write that function when **moving** data from C++ to Python is possible and
needed.

```c++
PyObject* Clif_PyObjFrom(std::shared_ptr<CppType>, py::PostProc);
```

Write that function when **sharing** data between C++ and Python is possible and
needed.

```c++
PyObject* Clif_PyObjFrom(CppType*, py::PostProc);
```

Write that function when **borrowing** data from C++ is needed. This is
extremely **dangerous** - Python will likely store the pointer that can easily
become dangling as C++ object lifetime is different from Python object lifetime.
Better use other conversion types.

### Python post processing

Sometimes a C++ type is not enough to determine which Python type to convert to.
For example in Python 3 `std::string` might be converted to `bytes` or `str`.
That information is provided in the .clif file and passed along in
`::clif::py::PostProc` argument. To get its definition
`#include "clif/python/postproc.h"`.

Postprocessing provides a function pointer to an `PyObject* (*)(PyObject*);` C
function that needs to be called during conversion to the Python type that needs
extra processing. However all other converter functions need to play along and
pass that information through even if they don't use it themselves.
That is especially true for containers to enable post-processing for contained
types. Take a look at `clif/python/stltypes.h` in the CLIF
runtime library and `util::StatusOr` example.

## Introducing a CLIF name for the C++ type

Usually you'll need a name to identify the C++ type within .clif files.
Yes, this name is internal to CLIF and works only inside `.clif` files. For
convenience CLIF names for standard types made the same as Python names
(eg. int, set, dict). Python knows nothing about CLIF names.

To add a CLIF name add a structured comment like

``` // CLIF use `::fq::CppType` as ClifName```

to your library header file.
The `ClifName` must be a valid Python name and be unique to CLIF (otherwise
you'll silently hide some other type from CLIF).

## Using the library

Besides adding the `cc_library` to the `py_clif_cc` deps use normal

```python
from "path/to/the/library/header" import *
```

to load the `ClifName` in a .clif file.


