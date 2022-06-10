

# RealTime-Video-Style-Transfer

A Style-Transfer modules for RealTime camera videos.




## Virtual Camera Creation

We use **pyvirtualcam** to create a virtual camera. 

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
Use this module to send processed images to the virtual camera.
```py
import test
test.run_virtual_camera('/dev/video2')
```