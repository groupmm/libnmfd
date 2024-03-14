# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))
LIBNMF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
assert os.path.exists(os.path.join(LIBNMF_DIR, 'libnmf'))
sys.path.insert(0, LIBNMF_DIR)
#sys.path.insert(0, os.path.join(LIBNMF_DIR, 'utils'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'libnmf'
copyright = '2024, Patricio López-Serrano, Christian Dittmar, Yigitcan Özer and Meinard Müller',
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx_rtd_theme',
			  'sphinx.ext.autodoc',  # documentation based on docstrings
              'sphinx.ext.napoleon',  # for having google/numpy style docstrings
              'sphinx.ext.viewcode',  # link source code
              'sphinx.ext.intersphinx',
              'sphinx.ext.autosummary',
              'sphinx.ext.extlinks']

templates_path = ['_templates']
exclude_patterns = []
# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_use_index = True
html_use_modindex = True
html_logo = os.path.join(html_static_path[0], 'libnmf_logo.png')
napoleon_custom_sections = [('Returns', 'params_style'), ('Parameters', 'params_style')]