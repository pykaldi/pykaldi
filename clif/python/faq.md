<!----- Conversion time: 2.025 seconds.
* GDC version 1.1.18 r6
* Fri Apr 07 2017 13:50:26 GMT-0700 (PDT)
* Source doc: http://go/clif-faq
----->


## CLIF - Frequently Asked Questions

[TOC]

---
This document answers some frequently asked questions about the CLIF
open source project. If you have a question that isn't answered here, join the
[discussion group](http://groups.google.com/group/pyclif) and ask away!

## General {#general}

#### **Q:** Specifying C++ type is annoying and unpythonic. Can we get rid of it? {#specifying-c-types}

**A:** We'd like to get rid of it, but it appeared to be difficult.

#### **Q:** PyCLIF describe types, why does it not follow [PEP 484](https://www.python.org/dev/peps/pep-0484/) style? {#not-pep-484}

**A:** PEP 484 was limited to (a) Python syntax and (b) Python execution
semantics.

PyCLIF is free of those limitations and chose the most expressive syntax for the
task.

#### **Q:** Can I wrap a class and specify a base class wrapped in another .clif file? {#base-class-in-another-clif-file}

**A:** No, that is not implemented.

#### **Q:** Why are there different quote types in CLIF? {#quote-types}

**A:** As you probably noticed, files are always in "double quotes" and C++
names in `backquotes`.

#### **Q:** What about interfacing C++ with languages other than Python such as Java and Go? {#other-languages}

**A:** CLIF is designed to support multiple language frontends. Python is our
first target. We expect success with Python to drive the desire to actually
implement and support other languages.


## SWIG {#swig}

#### **Q:** Can we still use CLIF and SWIG wrappers interchangeably? {#use-clif-and-swig}

**A:** CLIF can coexist with SWIG in one binary but no wrapped types can pass
between them.

#### **Q:** SWIG allows me to insert arbitrary Python / C++ code. Can I do the same in CLIF? {#arbitrary-code}

**A:** No. This SWIG misfeature proved to be difficult to maintain and
error-prone (almost no one gets Python refcounting and error processing right
especially for nested containers). All language code must go into corresponding
(py/c++) libraries for review and testing like any other code.  It also makes
such code available for static analysis and refactoring tools.

#### **Q:** SWIG supports overloaded C++ functions. How do I do that in CLIF? {#overloads}

**A:** Python does not support function overloading. To expose different
signatures of a C++ function to Python use different Python names (or default
arguments).

```
	def Func()
def `Func` as FuncWithDelay(delay: int)
```


## C++ types {#c-types}

#### **Q:** Why is my  __<put_your_favorite_here>__ C++ construct / API not supported? {#c-not-supported}

**A:** Not all C++ is supported by design. Plain C (macros, varargs, C arrays,
char\* strings) is not supported.

#### **Q:** How do I pass Callbacks (or other function pointers)? {#callbacks}

**A:** Use `std::function` instead. It will take any callable() but still check
number and type of arguments.

#### **Q:** Can I use the `T* output` convention to return additional values from a `std::function`? {#t*-output-with-std-function}

**A:** No. CLIF only supports input parameters in `std::function`.
Use `std::tuple` to return multiple values.

#### **Q:** Why doesn't CLIF accept varargs? {#no-varargs}

**A:** CLIF needs C++ type to be checked at language boundary and varargs does
not provide it.

#### **Q:** Why doesn't CLIF accept **T***? {#no-t*}

**A:** Many things in modern C++ are better passed by value e.g. `std::function`.

Also be extra careful accepting `T*` from CLIF - there is no object lifetime
guarantee, so don't store it.

#### **Q:** What about **const T***? {#const-t*}

**A:** Since Python doesn't have a good mapping of the C++ `const` construct
(there is no const object) CLIF doesn't support `const T*` other than making a
copy of it since it cannot guarantee object constness requested by C++ author.

#### **Q:** How to return a derived class pointer? My C++ function returns a base class pointer. {#derived-class-pointer}

**A:** Change your C++ signature to return the derived class pointer as it
actually does. It should not disrupt your C++ code (implicit cast will happen on
base class pointer assignment) but CLIF would know that a proper derived class
[smart] pointer is returned and do the right thing.

## Errors {#errors}

#### **Q:** I'm getting a compile error: no matching function for call to 'Clif_PyObjAs' {#no-matching-function-for-call-to-'clif_pyobjas'}

**A:** The .clif file declares a valid C++ type but does not import a CLIF
wrapper for that type, so CLIF does not know how to deal with it. Look at the
error - next lines show candidates with that type "<code>no known conversion
from <strong>'unknown_type</strong> *'</code>".
Use a [c_header_import](README.md?#cimport) statement to tell CLIF about this
type.

#### **Q:** I'm getting a compile error: call to deleted function 'Clif_PyObjFrom' {deleted-function-'clif_pyobjfrom'}

**A:** Python was trying create a copy on non-copyable C++ object. Either it was
an attempt of passing it by value or by const* (see [above](#const-t*)).
