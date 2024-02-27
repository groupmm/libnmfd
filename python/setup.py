"""
    Name: setup
    Date of Revision: Aug 2019
    Programmer: Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and  Meinard Müller

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    If you use the 'NMF toolbox' please refer to:
    [1]  Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and Meinard
        Müller
        NMF Toolbox: Music Processing Applications of Nonnegative Matrix
        Factorization
        In Proceedings of the International Conference on Digital Audio Effects
        (DAFx), 2019.

    License:
    This file is part of 'NMF toolbox'.
    https://www.audiolabs-erlangen.de/resources/MIR/NMFtoolbox/
    'NMF toolbox' is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    the Free Software Foundation, either version 3 of the License, or (at
    your option) any later version.

    'NMF toolbox' is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
    Public License for more details.

    You should have received a copy of the GNU General Public License along
    with 'NMF toolbox'. If not, see http://www.gnu.org/licenses/.
"""

from setuptools import setup, find_namespace_packages

setup(name='NMFtoolbox',
      description='NMF Toolbox: Music Processing Applications of Nonnegative Matrix Factorization',
      author='Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and  Meinard Müller',
      packages=find_namespace_packages(include=['NMFtoolbox.*']),
      install_requires=['numpy==1.16.1',
                        'scipy==1.2.1',
                        'matplotlib==3.1.1',
                        'jupyter==1.0.0',
                        'tqdm'],
      zip_safe=False)
