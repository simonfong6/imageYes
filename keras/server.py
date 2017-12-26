#!/usr/bin/env python
"""
Combines the youtube downloader and video classifier to download a video and 
classify it.
"""
from flask import Flask, request, send_file
from yt_classifier import YtClassifier

app = Flask(__name__)

@app.route('/')
def index():
    # Need dataset for label to name mapping ie. 001 --> ak47
    dataset_name = 'caltech'
    image_width,image_height = 299,299
    
    # Model to use to classify
    model_path = 'models/caltech_10_50_40_5_55.h5'

    y = YtClassifier(model_path, dataset_name, image_width, image_height)
    yt_url = "https://www.youtube.com/watch?v=G0uGB5C56Uk"
    output_video_name = "kimi_classified.mp4"
    y.classify(yt_url,output_video_name)
    return send_file(output_video_name)

@app.route('/video/<file_name>')
def video(file_name):
    output_video_name = "kimi_classified.mp4"
    return send_file(file_name)


def main():
    app.run(host='0.0.0.0', port=5000, threaded=True)
    

if(__name__ == '__main__'):
    main()
