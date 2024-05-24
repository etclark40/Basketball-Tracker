# Basketball-Tracker
Basketball Tracking Computer Vision Framework - Jetson Orin Nano

Ran on the Jetson Orin Nano developer kit using NVIDIA Jetpack 5.1.3 

# How To Run
First clone the repository,
```sh
git clone https://github.com/etclark40/Basketball-Tracker.git  
```
Then mount the directory into your existing inference container.  
```sh
cd jetson-inference  
docker/run.sh --volume ~/Basketball-Tracker:/Basketball-Tracker  
```
