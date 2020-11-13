# TFG

## Setup

```bash
conda create -n tfg python=3.8
conda install tensorflow-gpu==2.2.0
conda install pandas==1.1.2
pip install -r requirements.txt
conda install jupyterlab==2.2.6
conda install nodejs>=10
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

## Starting the project

```bash
jupyter lab
```

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda-repo-ubuntu2004-11-1-local_11.1.0-455.23.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-1-local_11.1.0-455.23.05-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-1-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda



- each patient has one or more scans (up to 26 scans)
- each scan has multiple slices = images (min 16, max 690)