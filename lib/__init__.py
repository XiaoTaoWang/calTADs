# Created on Wed Oct 29 16:55:39 2014

# Author: XiaoTao Wang
# Organization: HuaZhong Agricultural University

import xmlrpclib
from pkg_resources import parse_version as V

__author__ = 'XiaoTao Wang'
__version__ = '0.1.0-dev1'
__license__ = 'GPLv3+'

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