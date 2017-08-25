Developer's Conventions
***********************

While developing **pyKaldi**, we try our best to adhere to the following conventions:

Code
----

C++
~~~
.. todo:: add this part

Python
~~~~~~
.. todo:: add this part

Documentation
-------------
Python documentation follows Google Python Style Guide. Docstrings may extend over multiple lines.  Sections are created with a section header and a colon followed by a block of indented text. Sphinx extension, Napoleon, recognizes the following parameters in the docstrings:

- Args, Arguments, Params, Parameters
- Attributes
- Example(s)
- Note(s)
- Return(s)
- Raises
- See Also
- Todo
- Warning(s), Warns
- Yield(s)

For example,::

	def my_function(arg1, arg2):
		"""Computes the mean of arg1 and arg2, i.e. (arg1 + arg2) / 2

		Args:
			arg1 (int or float): Argument number 1
			arg2 (int or float): Argument number 2
			
		Returns:
			Mean of arg1 and arg2

		Example:
			>>> my_function(3, 5)
			>>> 4
			>>> my_function(10.0, 14.0)
			>>> 12.0
		"""