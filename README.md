# RealTime-Video-Style-Transfer

A Style-Transfer modules for RealTime camera videos.

---
## Preparation & Installation 

###
The trained-weights and data our system needs are stored [here](https://drive.google.com/drive/folders/1VJp29A73TzXNTAjESy8cvupRs4qRFRc3?usp=sharing). 
### Installation
The codes are run on **cuda=11.0, torch=1.9.0, torchvision=0.10.0**
```sh
conda create -n <env_name> python=3.8
conda install cudatoolkit==11.0.3 -c conda-forge
conda install pytorch==1.9.0 torchvision==0.10.0  cudatoolkit=11.0 -c pytorch -c conda-forge
pip install -r requirement.txt
```  
### Implementation

```sh

python test.py
```

---
## Virtual Camera Creation

We use **pyvirtualcam** to create a virtual camera if you want to use this module on Google Meet. 

The package works on Windows, macOS, and Linux, but we only test on Ubuntu.

### Python Package

Install **pyvirtualcam** from PyPI with:
```sh
pip install pyvirtualcam
```

### Ubuntu Package

**pyvirtualcam** uses **v4l2loopback** virtual cameras on Linux.

Install **v4l2loopback** on Ubuntu with:
```sh
sudo apt install v4l2loopback-dkms
```

### Usage
Create a virtual camera.
```sh
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback video_nr=2 card_label="Virtual Camera" exclusive_caps=1
```
To use this module to send processed images to the virtual camera.
```sh
python test.py --virtual_camera
```