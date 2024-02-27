"""
    Name: runTest
    Date of Revision: Jul 2019
    Programmer: Yiğitcan Özer

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    If you use the 'NMF toolbox' please refer to:
    []  Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and Meinard
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


from argparse import ArgumentParser
from NMFtoolbox.utils import run_matlab_script, get_matlab_matrices, run_unit_test
from __init__ import *

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-f', '--function_name',
                        type=str,
                        help='Function name to test.',
                        metavar='<FunctionName>')
    parser.add_argument('-m', '--matlab_path',
                        type=str,
                        help='Path to matlab binary file',
                        default='/usr/local/MATLAB/R2019a/bin/matlab',
                        metavar='<MatlabPath>')

    args = parser.parse_args()

    function_name = args.function_name
    path_to_matlab_bin = args.matlab_path

    run_matlab_script(function_name=function_name, path_to_matlab_bin=path_to_matlab_bin)

    matlab_dict = get_matlab_matrices(function_name=function_name)

    module = globals()['test_' + function_name]
    py_func_to_call = getattr(module, 'run_' + function_name)

    python_dict = py_func_to_call()

    print('\n################################################')
    print('Unit test runs for the function {}.'.format(function_name))
    print('################################################')

    for matrix_id in python_dict:
        print('\nRunning test for matrix {}'.format(matrix_id))
        matlab_mat = matlab_dict[matrix_id]
        python_arr = python_dict[matrix_id]

        run_unit_test(python_arr, matlab_mat, decimal_precision=4)
