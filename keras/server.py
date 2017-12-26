#!/usr/bin/env python
"""
Combines the youtube downloader and video classifier to download a video and 
classify it.
"""
from flask import Flask, request, send_from_directory, render_template
from yt_classifier import YtClassifier
import os

app = Flask(__name__)

VIDEO_DIR = "videos"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Need dataset for label to name mapping ie. 001 --> ak47
    dataset_name = 'caltech'
    image_width,image_height = 299,299
    
    # Model to use to classify
    model_path = 'models/caltech_10_50_40_5_55.h5'
    
    url = request.form['video_url']
    
    
    output_video_name = "out.mp4"
    video_path = os.path.join(VIDEO_DIR,output_video_name)
    
    y = YtClassifier(model_path, dataset_name, image_width, image_height)
    y.classify(url,video_path)
    return render_template('video.html', video=output_video_name)

@app.route('/videos/<video>')
def videos(video):
    return send_from_directory(VIDEO_DIR,video)


def main():
    app.run(host='0.0.0.0', port=5000, threaded=True)
    

if(__name__ == '__main__'):
    main()
