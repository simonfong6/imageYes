#!/env/bin/python
"""
COGS 118B Project
Author: Simon Fong, Thinh Le, Wilson Tran

Example Run Command:
python live.py video_path model_path output_video_name flag_for_showing_image

python live.py kimi.mp4 model.h5 kimi_out.mp4 False
"""

from keras.models import Model, load_model
import numpy as np
import os
import sys
import cv2
from dataset import Dataset                     # Custom Dataset

class VideoClassifier:
    """Class to load a video and classify each frame of a video"""
    
    def __init__(self,input_video_name, dataset_name, image_height, 
        image_width, model_name, output_video_name, show_image=False):
        """Setup input video stream, model, and output video stream """
        
        # Define VideoCapture object
        if(input_video_name == '0'):
            # 0: Built in webcam
            self.input_video = cv2.VideoCapture(0)       
        elif(input_video_name == '1'):
            # 1: External webcam
            self.input_video = cv2.VideoCapture(1)       
        else:
             # If not webcam, the open video
            self.input_video = cv2.VideoCapture(input_video_name) 
        
        # Load dataset
        dataset = Dataset(dataset_name,image_height,image_width)
        dataset.read_data()        
        self.dataset_map = dataset.names
        self.image_height = image_height
        self.image_width = image_width
        
        # Make model
        print("Loading model from {}".format(model_name))
        self.model = load_model(model_name)
        print("{} loaded".format(model_name))
        
        # Save output video name
        self.output_video_name = output_video_name
        
        # Define the codec
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # Get videocapture's shape
        out_shape = (int(self.input_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # Set FPS from video file
        fps = self.input_video.get(cv2.CAP_PROP_FPS)

        # Create VideoWriter object
        self.output_video = cv2.VideoWriter(self.output_video_name, 
            fourcc, fps, out_shape)    
        
        # Flag for whether or not to show images while processing
        self.SHOW_IMAGE = show_image
        
        # For progress spinner
        self.iterations = 0
    
    def get_frame(self):
        """Takes a frame from the video and returns it in the proper format"""
        
        # Capture frame-by-frame
        ret, original_image = self.input_video.read()
        
        # Check if frame was captured
        if(ret == False):
            raise TypeError("Image failed to capture")

        # Creates an image list because the model expects 4 dimensions
        images = []

        # Resize image
        image_resized = cv2.resize(original_image, (self.image_height,
                            self.image_width))
        images.append(image_resized)
        
        # Convert list to numpy array
        images = np.array(images)

        return (images, original_image)
    
    def release(self):
        """Release and destory everything"""
        
        self.input_video.release()
        self.output_video.release()
        cv2.destroyAllWindows()
        print("Finished processing video, saving file to {}".format(
            self.output_video_name))
    
    def classify(self):
        """Classify all the frames of the video and save the labeled video"""
        
        # Continuously grab a frame from the camera and classify it
        print("Classifying the video")
        while(self.input_video.isOpened()):
            self.spin()
            try:
                # Capture image from webcam
                images, original_image = self.get_frame()
            
            except TypeError, e:
                # Break if image failed to capture
                print(e)
                break
            
            # Classify image
            label = self.model.predict(images)
            label = self.dataset_map[np.argmax(label[0])].split('.')[1]
            
            #print(label)
            
            # Print the text on the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(original_image,label,(50,50), font, 
                        1,(255,255,255),2,cv2.LINE_AA)
            
            # Write image to video
            self.output_video.write(original_image)
            
            
            # Show the image if flag is set
            if(self.SHOW_IMAGE):
                cv2.imshow('image',original_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        self.release()
        
    def spin(self):
        "Spin the progress spinner"
        self.iterations += 1
        spin_states = {
                        0: "-",
                        1: "\\",
                        2: "|",
                        3: "/",
                        4: "-",
                        5: "\\",
                        6: "|",
                        7: "/",
                        }
        state = spin_states[self.iterations%8]
        sys.stdout.write(state + "\r")
        sys.stdout.flush()
                   
def main():
    
    # Need dataset for label to name mapping ie. 001 --> ak47
    dataset_name = 'caltech'
    image_width,image_height,num_channels = 299,299,3

    # Get input video from command line
    input_video_name = str(sys.argv[1])
    
    # Get model from command line
    model_name = str(sys.argv[2])

    # Get output video name from command line
    output_video_name = str(sys.argv[3])

    # Get show image flag from command line
    show_image = sys.argv[4] == 'True'

    video_classifier = VideoClassifier(input_video_name,dataset_name,
                            image_height, image_width, model_name,          
                            output_video_name, show_image)
    
    try:
        video_classifier.classify()
    
    except KeyboardInterrupt:
        # Release and destory everything
        video_classifier.release()
        print("Exiting on Interrupt")

if __name__ == '__main__':
    main()
