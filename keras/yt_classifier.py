#!/usr/bin/env python
"""
Combines the youtube downloader and video classifier to download a video and 
classify it.
"""
from video_classifier import VideoClassifier
from downloader import Downloader

class YtClassifier:
    
    def __init__(self, model_path, dataset_name, image_width, image_height):
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.image_width = image_width
        self.image_height = image_height
    
    def classify(self, yt_url, output_video_name):
        
        d = Downloader(yt_url)
        input_video_name = d.download()
        
        v = VideoClassifier(
            input_video_name,self.dataset_name,
            self.image_height, self.image_width,
            self.model_path, output_video_name)
            
        v.classify()
        d.remove()

def main():
    # Need dataset for label to name mapping ie. 001 --> ak47
    dataset_name = 'caltech'
    image_width,image_height = 299,299
    
    # Model to use to classify
    model_path = 'models/caltech_10_50_40_5_55.h5'

    y = YtClassifier(model_path, dataset_name, image_width, image_height)
    yt_url = "https://www.youtube.com/watch?v=G0uGB5C56Uk"
    output_video_name = "kimi_classified.mp4"
    y.classify(yt_url,output_video_name)

if(__name__ == '__main__'):
    main()
