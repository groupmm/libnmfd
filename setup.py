from setuptools import setup, find_packages


with open('README.md', 'r') as fdesc:
    long_description = fdesc.read()

setup(
    name='libnmfd',
    version='1.0.0',
    description='libnmfd: Music Processing Applications of Nonnegative Matrix Factorization',
    author='Patricio López-Serrano, Christian Dittmar, Yigitcan Özer and Meinard Müller',
    author_email='christian.dittmar@iis.fraunhofer.de',
    url='https://github.com/groupmm/libnmfd',
    download_url='https://github.com/groupmm/libnmfd',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 3",
    ],
    keywords='audio music sound nmf decomposition',
    license='MIT',
    install_requires=['ipython >= 7.10.0, < 8.0.0',
                      'jupyter >= 1.0.0',
                      'matplotlib >= 3.1.0, < 4.0.0',
                      'numpy >= 1.17.0, < 2.0.0',
                      'pandas >= 1.0.0, < 2.0.0',
                      'pysoundfile >= 0.9.0, < 1.0.0',
                      'scipy >= 1.7.0, < 2.0.0',
                      'tqdm >= 4.0.0'],
    python_requires='>=3.8, <4.0',
    extras_require={
        'tests': ['pytest == 6.2.*'],
        'docs': ['sphinx == 4.0.*',
                 'sphinx_rtd_theme == 0.5.*']
    }
)