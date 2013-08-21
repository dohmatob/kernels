Kernels is a project on kernel methods (spectral string kernels, string mismatch kernels).
It contains 2 implmentations: C++ and Pythonic.

Dependencies (only for C++ version)
===================================
* cmake (the build tool)
* doxygen (optional, except if you need to generate documentation)
* boost (for the algebra invoked)


Build (only C++ version)
===================
          cmake . && make all

Examples
========

Python example
---------------
To run a python example, type:

          python python/trie.py

C++ example
-----------
To run the a C++ example, type:

          ./bin/main 2 4 0 data/dummy_data.txt data/dummy_kernel.txt

Testing
=======

Python tests
------------
To run the python tests (you'll need nosetests test tool), type:

          nosetests -v python/trie.py

C++ tests
---------
To run the C++ tests (you'll need boost component unit_test_framework), type:

          ./bin/test
          
(c) DOHMATOB Elvis DOpgima
