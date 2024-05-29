# Basketball Shot Detector
Developed on the Jetson Orin Nano using NVIDIA Jetpack 5.1.3   

![Basketball Orin Nano Demo](https://github.com/etclark40/Basketball-Tracker/assets/131305180/71048368-2382-42e6-bfe8-9ef6f4ae37ea)

# How To Run
First clone the repository.
```sh
git clone https://github.com/etclark40/Basketball-Tracker.git  
```
Then run an inference container with the directory mounted.
```sh
cd jetson-inference  
docker/run.sh --volume ~/Basketball-Tracker:/Basketball-Tracker  
```
To run the application:
```sh
python3 runTracker.py --model=ssd/ssd-mobilenet.onnx --labels=ssd/labels.txt --input-blob=input_0 --output-cvg=scores --output-bbox=boxes /dev/video0
```
