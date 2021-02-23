#!/bin/sh

# Empty PYTHONPATH
export PYTHONPATH=

# Download and install miniconda
MINICONDA_INSTALLER_SCRIPT=Miniconda3-4.5.4-Linux-x86_64.sh
MINICONDA_PREFIX=/usr/local
wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT
chmod +x $MINICONDA_INSTALLER_SCRIPT
./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX


# Install Python3.7 with miniconda and update all pre-installed packages to latest versions
conda install --channel defaults conda python=3.7 --yes
conda update --channel defaults --all --yes


# Install Cusignal with conda
conda install -c rapidsai -c conda-forge cusignal cudatoolkit=10.1 --yes


# Update Copy with pip
pip3 install cupy-cuda101==8.1.0 --upgrade

# Install pyFFTW and PyLops
pip3 install pyFFTW
conda install -c conda-forge pylops  --yes

# Remove cupy from original Colab installation to avoid conflicts
rm -rf /usr/local/lib/python3.7/dist-packages/cupy*
