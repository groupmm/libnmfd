.. libnmfd documentation master file, created by
   sphinx-quickstart on Tue Aug 29 10:47:22 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

libnmfd: Music Processing Applications of Nonnegative Matrix Factorization
==========================================================================

Nonnegative matrix factorization (NMF) is a family of methods widely used for information retrieval across domains
including text, images, and audio. Within music processing, NMF has been used for tasks such as transcription,
source separation, and structure analysis. Prior work has shown that initialization and constrained update rules can
drastically improve the chances of NMF converging to a musically meaningful solution. Along these lines we present the
libnmfd (NMF toolbox), which contains Python implementations of conceptually distinct NMF variants---in particular,
the repository includes an overview for two algorithms. The first variant, called nonnegative matrix factor
deconvolution (NMFD), extends the original NMF algorithm to the convolutive case, enforcing the temporal order of
spectral templates. The second variant, called diagonal NMF, supports the development of sparse diagonal structures in
the activation matrix. Our toolbox contains several demo applications and code examples to illustrate its potential and
functionality. See also https://www.audiolabs-erlangen.de/resources/MIR/NMFtoolbox.


.. toctree::
   :maxdepth: 2
   :caption: API Documentation
   :hidden:

   core/index
   dsp/index
   utils/index

.. toctree::
   :caption: Reference
   :maxdepth: 1
   :hidden:

   genindex
   py-modindex

