#!/usr/bin/env python
"""
Combines the youtube downloader and video classifier to download a video and 
classify it.
"""
from video_classifier import VideoClassifier
from downloader import Downloader

class YtClassifier:
    
    def __init__(self, dataset_name, image_height, image_width, model_name):
        self.video_classifier = VideoClassifier(dataset_name, 
            image_height, image_width, model_name)
    
    def classify(self, url, output_video_name):
        
        d = Downloader(url)
        input_video_name = d.download()
        self.video_classifier.classify(input_video_name, output_video_name)
        d.remove()

def main():
    # Need dataset for label to name mapping ie. 001 --> ak47
    dataset_name = 'caltech'
    image_width,image_height = 299,299
    
    # Model to use to classify
    model_path = 'models/caltech_10_50_40_5_55.h5'

    y = YtClassifier(dataset_name, image_height, image_width, model_name)
    url = "https://www.youtube.com/watch?v=G0uGB5C56Uk"
    output_video_name = "bunny.mp4"
    y.classify(url,output_video_name)

if(__name__ == '__main__'):
    main()
