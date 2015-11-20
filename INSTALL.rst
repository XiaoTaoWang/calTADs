Installation Guide for calTADs
==============================

Requirements
============
calTADs is developed and tested on UNIX-like operating system, and following Python
packages are recommended:

- Python (2.x >= 2.6, not compatible with 3.x)
- Numpy (>= 1.6)
- Scipy library (>= 0.10)
- ghmm (>= 0.9)

.. note:: Tested systems: Red Hat Enterprise Linux Server release 6.4 (Santiago)

Installation
=============
Firstly, use `conda <http://conda.pydata.org/miniconda.html>`_ to manage Python
environment and install *numpy* and *scipy*::

    $ conda install numpy scipy

The General Hidden Markov Model library (`GHMM <http://sourceforge.net/projects/ghmm/>`_)
is originally developed in *C*, which is not maintained at Binstar or PyPI, and has to
be installed from source code.

Download the source code, unpack it, change to the extracted directory, get to a
terminal prompt and type::

    $ ./configure --prefix=GHMM_HOME
    $ make
    $ make check
    $ make install

where GHMM_HOME is the installation directory you choose for GHMM.

.. note:: You may run into such error "/bin/rm cannot remove 'libtoolT': no such file
   or directory" when you configure GHMM for your system, then you can try to edit
   the configure file and remove the line containing ``$RM "$cfgfile"``, and configure
   GHMM again.

You need to reset your LD_LIBRARY_PATH environment variables then. Use vi command
to add this line to ``~``/.bashrc::

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:GHMM_HOME/lib

To update the environment variables::

    $ source ~/.bashrc

Then it's time to install the Python wrapper for GHMM. Change to the *ghmmwrapper*
base directory::

    $ swig -noruntime -python -nodefault ghmmwrapper.i
    $ python setup.py build
    $ python setup.py install

Now, you can install calTADs using *easy_install*::

    $ conda install setuptools
    $ easy_install calTADs

Enjoy it!

