# Created on Thu Nov 19 22:37:15 2015

# Author: XiaoTao Wang
# Organization: HuaZhong Agricultural University

"""
This is a free software under GPLv3. Therefore, you can modify, redistribute
or even mix it with other GPL-compatible codes. See the file LICENSE
included with the distribution for more details.

"""
import os, sys, lib
from distutils.core import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

if (sys.version_info.major != 2) or (sys.version_info.minor < 6):
    print 'PYTHON VERSION MUST BE 2.6 or 2.7. YOU ARE CURRENTLY USING PYTHON ' + sys.version
    sys.exit(2)

# Guarantee Unix Format
text = open('script/calTADs', 'rb').read().replace('\r\n', '\n')
open('script/calTADs', 'wb').write(text)

setup(
    name = 'calTADs',
    version = lib.__version__,
    author = lib.__author__,
    author_email = 'wangxiaotao868@163.com',
    url = 'https://github.com/XiaoTaoWang/calTADs/',
    description = 'A hierarchical domain caller for Hi-C data based on a modified version of Directionality Index',
    keywords = 'Hi-C TAD DI directionality index topologically associating domain chromosome organization',
    package_dir = {'calTADs':'lib'},
    packages = ['calTADs'],
    scripts = ['script/calTADs'],
    long_description = lib.long_description,
    classifiers = [
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: POSIX',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        ]
    )
