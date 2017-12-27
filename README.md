# ImageYes
Comparison of using transfer learning on the Caltech 256 dataset and on an unclean dataset scraped from the web.This is for COGS 118B final project.  

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
Head to [localhost:5000] and enter a video URL such as this one, https://www.youtube.com/watch?v=lTTajzrSkCw . After submitting, it will take some time to process. Eventually, you will see the video specified with classification labels.
# Other Useful Resources

## Installing PyCharm Community Edition by PPA
[https://itsfoss.com/install-pycharm-ubuntu/]  
Add the PPA
```
sudo add-apt-repository ppa:mystic-mirage/pycharm
sudo apt-get update
```
Install Pycharm Community Edition
```
sudo apt-get install pycharm-community
```


# Resources
[Caltech 256 Dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech256/)  
[Google Image Webscraper](https://github.com/hardikvasa/google-images-download)  
[Converting AVI to MP4 in Python](https://stackoverflow.com/questions/22748617/python-avi-to-mp4)  
[subprocess.call](http://www.pythonforbeginners.com/os/subprocess-for-system-administrators)  
[Keras and Flask Bug Solution](https://stackoverflow.com/questions/43822458/loading-a-huge-keras-model-into-a-flask-app/47991642#47991642)  
Cannot run flask as ```threaded=True``` or else it will break the Keras predict method. Unsure why
[Pycharm Installation](https://itsfoss.com/install-pycharm-ubuntu/)  
