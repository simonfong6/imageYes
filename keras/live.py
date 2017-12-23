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

# Need dataset for label to name mapping ie. 001 --> ak47
DATASET_NAME = 'caltech'
IMAGE_WIDTH,IMAGE_HEIGHT,NUM_CHANNELS = 299,299,3

# Command line arg for choosing model
model_path = str(sys.argv[2])

# Load dataset
cal = Dataset(DATASET_NAME,IMAGE_HEIGHT,IMAGE_WIDTH)
cal.read_data()
names = cal.names


# Define VideoCapture object
cap = None
camera = str(sys.argv[1])

# Define output video shape
out_shape = None

# Flag to show image during runtime
SHOW_IMAGE = sys.argv[4] == 'True'

if(camera == '0'):
    cap = cv2.VideoCapture(0)       # 0: Built in webcam
elif(camera == '1'):
    cap = cv2.VideoCapture(1)       # 1: External webcam
else:
    cap = cv2.VideoCapture(camera)  # If not webcam, then open video

# Define the codec
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Get output video name from command line
out_name = str(sys.argv[3])

# Get videocapture's shape
out_shape = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# Set FPS from video file
FPS = cap.get(cv2.CAP_PROP_FPS)

# Create VideoWriter object
out = cv2.VideoWriter(out_name,fourcc,FPS,out_shape)
        
def take_picture():
    """Takes a picture and returns it in the proper format"""
    
    # Capture frame-by-frame
    ret, original_image = cap.read()
    
    # Check if frame was captured
    if(ret == False):
        raise TypeError("Image failed to capture")

    # Creates an image list because the model expects 4 dimensions
    images = []

    # Resize image
    image_resized = cv2.resize(original_image, (IMAGE_HEIGHT,IMAGE_WIDTH))
    images.append(image_resized)
    
    # Convert list to numpy array
    images = np.array(images)

    return (images, original_image)


def main():
    
    # Make model
    model = load_model(model_path)
    print("{} loaded".format(model_path))
    
    try:
        # Continuously grab a frame from the camera and classify it
        while(cap.isOpened()):
            
            try:
                # Capture image from webcam
                images, original_image = take_picture()
            
            except TypeError, e:
                # Break if image failed to capture
                print(e)
                break
            
            # Classify image
            label = model.predict(images)
            label = names[np.argmax(label[0])].split('.')[1]
            
            #print(label)
            
            # Print the text on the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(original_image,label,(50,50), font, 1,(255,255,255),2,cv2.LINE_AA)
            
            # Write image to video
            out.write(original_image)
            
            
            # Show the image if flag is set
            if(SHOW_IMAGE):
                cv2.imshow('image',original_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        # Release and destory everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Finished processing video, saving file to {}".format(out_name))
    
    except KeyboardInterrupt:
        # Release and destory everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Exiting on Interrupt, saving file to {}".format(out_name))



if __name__ == '__main__':
    main()
