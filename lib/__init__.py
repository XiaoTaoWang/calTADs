# Created on Wed Oct 29 16:55:39 2014

# Author: XiaoTao Wang
# Organization: HuaZhong Agricultural University

import xmlrpclib
from pkg_resources import parse_version as V

__author__ = 'XiaoTao Wang'
__version__ = '0.1.0-dev1'
__license__ = 'GPLv3+'

long_description = """
Introduction
============
3C-based techniques(5C, Hi-C) have revealed the existence of topologically
associating domains(TADs), a pervasive sub-megabase scale structure of chromosome.
TADs are contiguous regions in which loci interact much more frequently with
each other than with loci out of the region. Visually, TADs appear as square
blocks along the diagonal on a heatmap.

There are various methods for TAD identification [1]_, [2]_. Most methods
apply a two-step scheme: First, transform TAD or boundary signal into 1d
profile using some statistic(e.g. Directionality Index, DI); Then, use the
1d profile to identify potential boundaries and produce a set of discrete
non-overlapping TADs. However, the organization of chromosome structure is
always intricate and hierarchical. Phillips-Cremins JE et al. [3]_ utilized
a modified DI of multiple scales subdivided TADs into smaller subtopologies (sub-TADs)
using 5C data. Here, I extend their algorithm to the whole genome and develop
this software.

*calTADs* are tested on traditional [4]_ and *in-situ* [5]_ Hi-C data, both generating
reasonable results.

Installation
============
Please check the file "INSTALL.rst" in the distribution.

Links
=====
- `Repository <https://github.com/XiaoTaoWang/calTADs>`_
- `PyPI <https://pypi.python.org/pypi/calTADs>`_

Usage
=====
Open a terminal, type ``calTADs -h`` for help information.

calTADs contains a process management system, so you can submit the same
command repeatedly to utilize the parallel power as much as possible.

Reference
=========
.. [1] Dixon JR, Selvaraj S, Yue F et al. Topological domains in
   mammalian genomes identified by analysis of chromatin interactions.
   Nature, 2012, 485: 376-380.

.. [2] Sexton T, Yaffe E, Kenigsberg E et al. Three-dimensional folding
   and functional organization principles of the Drosophila genome.
   Cell, 2012, 148: 458-472.

.. [3] Phillips-Cremins JE, Sauria ME, Sanyal A et al. Architectural protein
   subclasses shape 3D organization of genomes during lineage commitment.
   Cell, 2013, 153(6):1281-95.

.. [4] Lieberman-Aiden E, van Berkum NL, Williams L et al. Comprehensive
   mapping of long-range interactions reveals folding principles of the
   human genome. Science, 2009, 326: 289-293.

.. [5] Rao SS, Huntley MH, Durand NC. A 3D map of the human genome at
   kilobase resolution reveals principles of chromatin looping.
   Cell, 2014, 159(7):1665-80.
"""

## Check for update
try:
    pypi = xmlrpclib.ServerProxy('http://pypi.python.org/pypi')
    available = pypi.package_releases('calTADs')
    if V(__version__) < V(available[0]):
        print '*'*75
        print 'Version %s is out of date, Version %s is available.' % (__version__, available[0])
        print 'Run `pip install -U calTADs` or `easy_install -U calTADs` for update.'
        print
        print '*'*75
except:
    pass

Me = __file__