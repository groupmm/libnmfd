<table border="0">
  <tr>
    <td><img src=docs/build/html/_static/libnmfd_logo.png alt="libsoni logo" width="1000"></td>
    <td><h2>libnmfd: Music Processing Applications of Nonnegative Matrix Factorization</h2>
<br> <br>
</td>
  </tr>
</table>

Nonnegative matrix factorization (NMF) is a family of methods widely used for information retrieval across domains
including text, images, and audio. Within music processing, NMF has been used for tasks such as transcription,
source separation, and structure analysis. Prior work has shown that initialization and constrained update rules can 
drastically improve the chances of NMF converging to a musically meaningful solution. Along these lines we present the 
libnmfd (NMF toolbox), which contains Python implementations of conceptually distinct NMF variants---in particular,
the repository includes an overview for two algorithms. The first variant, called nonnegative matrix factor
deconvolution (NMFD), extends the original NMF algorithm to the convolutive case, enforcing the temporal order of 
spectral templates. The second variant, called diagonal NMF, supports the development of sparse diagonal structures in 
the activation matrix. Our toolbox contains several demo applications and code examples to illustrate its potential and 
functionality. See also the [AudioLabs webpage](https://www.audiolabs-erlangen.de/resources/MIR/NMFtoolbox).

## Installation Guide
We outline two primary methods for setting up ``libnmfd`` using pip and setting up a dedicated environment.

### Method I: Installing with pip
Utilize Python's package manager, pip, for a straightforward installation of ``libnmfd``:

```
pip install libnmfd
```
Note: We advise performing this installation within a Python environment (such as conda or a virtual environment) 
to prevent any conflicts with other packages. Ensure your environment runs Python 3.7 or higher.

### Method II: Setting Up a Conda Environment
Alternatively, you can establish a conda environment specifically for ``libnmfd`` by employing the 
``environment_libnmfd.yml`` file. This approach not only installs ``libnmfd`` but also includes necessary packages like
libnmfd and jupyter to facilitate running demo files. Run the following command:


```
conda env create -f environment_libnmfd.yml
```


## Running Example Notebooks
To explore ``libnmfd`` through example notebooks:

1. **Install ``libnmfd``:** Prior to cloning the repository and running the notebooks, ensure libnmfd and its dependencies are installed (as described above).
2. **Clone the repository:** Download the ``libnmfd`` repository to your local machine using the following git command:
   
```
git clone https://github.com/groupmm/libnmfd.git
```

3. **Install Jupyter:** If not already installed via the conda environment setup, install Jupyter to run the notebooks:

```
pip install jupyter
```

4. **Launch Jupyter Notebook:** Start the Jupyter notebook server by executing: 
```
jupyter notebook
```
This will open a browser window from where you can navigate to and open the example notebooks.


## Licence
The code for this toolbox is published under an [MIT licence](LICENCE).

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
