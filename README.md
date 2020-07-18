# Fake background pytorch

### TODO
- should work for different cameras

### How to run this?

Install dependencies for [pyfakewebcam](https://github.com/jremmons/pyfakewebcam)

```bash
# Create virtual camera using:
modprobe v4l2loopback devices=1
# Run
python fakecam.py --device '/dev/video2' --background ./img.jpg
```
