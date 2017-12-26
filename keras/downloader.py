#!/usr/bin/env python
"""
Downloads specified video from YouTube.
"""

from __future__ import unicode_literals
import youtube_dl
import os

class Downloader:
    
    def __init__(self,url, temp_dir='temp', temp_video='temp'):
        self.url = url
        self.TEMP_DIR = temp_dir
        self.TEMP_VIDEO = temp_video
        
    def download(self):
        """Downloads the video from specified by the url returns its path"""
        
        # If the temp dir doesn't exist, create it
        if(not os.path.isdir(self.TEMP_DIR)):
            os.mkdir(self.TEMP_DIR)
        
        # Formats url into a list of urls for the downloader
        urls = [self.url]

        # Downloader options: Save video in the temp dir
        ydl_opts = {
                    'format': 'bestvideo+bestaudio/best',
                    'outtmpl': unicode(os.path.join(self.TEMP_DIR,
                                    self.TEMP_VIDEO))
        }

        # Download video
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download(urls)

        # List all the files in temp dir, should only be the temp video       
        temp_file_names = os.listdir(self.TEMP_DIR)
        
        # Return the path to the file
        self.temp_file_path = os.path.join(self.TEMP_DIR,temp_file_names[0])
        
        return self.temp_file_path
        
    def remove(self):
        """Deletes the downloaded youtube video."""
        os.remove(self.temp_file_path)

        
def main():
    d = Downloader("https://www.youtube.com/watch?v=a2GujJZfXpg")
    
    print(d.download())
    d.remove()
    
if(__name__ == '__main__'):
    main()
