# ImageYes
Comparison of using transfer learning on the Caltech 256 dataset and on an unclean dataset scraped from the web.This is for COGS 118B final project.  

# Software Dependencies
Install keras, tensorflow, h5py for model inferencing
Install flask for web server component
```
pip install keras tensorflow h5py flask youtube_dl
```
Installing OpenCV
```
wget https://raw.githubusercontent.com/milq/milq/master/scripts/bash/install-opencv.sh  
sudo bash install-opencv.sh
```
Symbolicallyy link the OpenCV library to python libraries (Unsure why this works, but needed to do this before creating a virtualenv and using OpenCV. OpenCV will work outside a virtualenv without this.)
```
sudo ln -s /usr/local/lib/python2.7/dist-packages/cv2.so  /usr/lib/python2.7/
```

# Webscraper

# Experiment Results

# Video Classifier
video_classifier.py  
Takes in a video stream, classifies each frame, and creates a new video with classification captions.
## Built-In Webcam
Running using built-in webcam without showing the video while processing. If you want to see video while it's classifying you can set the last argument to True.
```
python video_classifier.py 0 model.h5 out.mp4 False
```

## External Webcam
```
python video_classifier.py 1 model.h5 out.mp4 False
```

## Video File
```
python video_classifier.py video.mp4 model.h5 out.mp4 False
```

# Video Downloader

# Video Classifier by URL

# Video Classifier Web Application
Provides form to enter a URL to a video and returns that video with classification captions.
## Execution
```
python server.py
```
Head to http://localhost:5000/ and enter a video URL such as this one, https://www.youtube.com/watch?v=lTTajzrSkCw . After submitting, it will take some time to process. Eventually, you will see the video specified with classification labels.
# Other Useful Resources

## Installing PyCharm Community Edition by PPA
https://itsfoss.com/install-pycharm-ubuntu/  
Add the PPA
```
sudo add-apt-repository ppa:mystic-mirage/pycharm
sudo apt-get update
```
Install Pycharm Community Edition
```
sudo apt-get install pycharm-community
```

## Installing Atom
Add the PPA
```
sudo add-apt-repository ppa:webupd8team/atom
sudo apt update
```
Install Atom
```
sudo apt install atom
```

## Adding swap space
```
sudo /bin/dd if=/dev/zero of=/var/swap.1 bs=1M count=1024  
sudo /sbin/mkswap /var/swap.1  
sudo /sbin/swapon /var/swap.1  
```

## Removing swap space
```
sudo swapoff /var/swap.1
sudo rm /var/swap.1
```

## Runtime Notes
Runtime 27 secs for 11 sec video on Laptop at 2.45 runtime secs/video secs  
Runtime 85 secs for 18 sec video on Laptop at 4.72 runtime secs/video secs  

# Resources
[Caltech 256 Dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech256/)  
[Google Image Webscraper](https://github.com/hardikvasa/google-images-download)  
[Converting AVI to MP4 in Python](https://stackoverflow.com/questions/22748617/python-avi-to-mp4)  
[subprocess.call](http://www.pythonforbeginners.com/os/subprocess-for-system-administrators)  
[Keras and Flask Bug Solution](https://stackoverflow.com/questions/43822458/loading-a-huge-keras-model-into-a-flask-app/47991642#47991642)  
Cannot run flask as ```threaded=True``` or else it will break the Keras predict method. Unsure why  
[Pycharm Installation](https://itsfoss.com/install-pycharm-ubuntu/)  
[OpenCV Install](http://milq.github.io/install-opencv-ubuntu-debian/)  
[Swapspace for Keras Installation](https://stackoverflow.com/questions/19595944/trouble-installing-scipy-in-virtualenv-on-a-amazon-ec2-linux-micro-instance)  
[10 Ways to Host Web Applications](https://blog.patricktriest.com/host-webapps-free/)  
[Installing Atom](http://tipsonubuntu.com/2016/08/05/install-atom-text-editor-ubuntu-16-04/)  
[POSSIBLE OpenCV on Lambda](https://github.com/aeddi/aws-lambda-python-opencv)  
[POSSIBLE Keras on Lambda](https://github.com/sunilmallya/keras-lambda)  
