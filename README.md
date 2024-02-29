## libnmf

Nonnegative matrix factorization (NMF) is a family of methods widely used for information retrieval across domains
including text, images, and audio. Within music processing, NMF has been used for tasks such as transcription,
source separation, and structure analysis. Prior work has shown that initialization and constrained update rules can 
drastically improve the chances of NMF converging to a musically meaningful solution. Along these lines we present the 
libnmf (NMF toolbox), containing MATLAB and Python implementations of conceptually distinct NMF variants---in particular,
the repository includes an overview for two algorithms. The first variant, called nonnegative matrix factor
deconvolution (NMFD), extends the original NMF algorithm to the convolutive case, enforcing the temporal order of 
spectral templates. The second variant, called diagonal NMF, supports the development of sparse diagonal structures in 
the activation matrix. Our toolbox contains several demo applications and code examples to illustrate its potential and 
functionality. By providing MATLAB and Python code on a documentation website under a GNU-GPL license, as well as 
including illustrative examples, our aim is to foster research and education in the field of music processing.

See also the [AudioLabs webpage](https://www.audiolabs-erlangen.de/resources/MIR/NMFtoolbox).

## References

[1] Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and Meinard Müller\
**NMF Toolbox: Music Processing Applications of Nonnegative Matrix Factorization** \
In Proceedings of the International Conference on Digital Audio Effects (DAFx), 2019.

[2] Christian Dittmar and Meinard Müller \
**Reverse Engineering the Amen Break — Score-Informed Separation and Restoration Applied to Drum Recordings** \
IEEE/ACM Transactions on Audio, Speech, and Language Processing, 24(9): 1531–1543, 2016. 

[3] Jonathan Driedger, Thomas Prätzlich, and Meinard Müller \
**Let It Bee — Towards NMF-Inspired Audio Mosaicing** \
In Proceedings of the International Society for Music Information Retrieval Conference (ISMIR): 350–356, 2015. 

[4] Paris Smaragdis \
**Non-negative Matrix Factor Deconvolution; Extraction of Multiple Sound Sources from Monophonic Inputs** \
In Proceedings of the International Conference on Independent Component Analysis and Blind Signal Separation 
(ICA): 494–499, 2004.

[5] Daniel D. Lee and H. Sebastian Seung \
**Learning the parts of objects by non-negative matrix factorization** \
Nature, 401(6755): 788–791, 1999. 