#!/env/bin/python
"""
COGS 118B Project
Author: Simon Fong, Thinh Le, Wilson Tran
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

if(camera == '0'):
    cap = cv2.VideoCapture(0)       # 0: Built in webcam
elif(camera == '1'):
    cap = cv2.VideoCapture(1)       # 1: External webcam
else:
    cap = cv2.VideoCapture(camera)  # If not webcam, then open video
    
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_name = str(sys.argv[3])
out = cv2.VideoWriter(out_name,fourcc, 20.0,
                     (2*IMAGE_HEIGHT,2*IMAGE_WIDTH))
        
def take_picture():
    """Takes a picture and returns it in the proper format"""
    
    # Capture frame-by-frame
    ret, image = cap.read()
    
    # Check if frame was captured
    if(ret == False):
        return None

    # Creates an image list because the model expects 4 dimensions
    images = []

    # Resize image
    image_resized = cv2.resize(image, (IMAGE_HEIGHT,IMAGE_WIDTH))
    images.append(image_resized)
    
    # Convert list to numpy array
    images = np.array(images)

    return images


def main():
    
    # Make model
    model = load_model(model_path)
    print("{} loaded".format(model_path))
    
    # Continuously grab a frame from the camera and classify it
    while(cap.isOpened()):
        # Capture image from webcam
        images = take_picture()
        
        # Break if image failed to capture
        if(images is None):
            break
        
        # Classify image
        label = model.predict(images)
        label = names[np.argmax(label[0])].split('.')[1]
        
        print(label)
        
        # Print the text on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        image = cv2.resize(images[0], (2*IMAGE_HEIGHT,2*IMAGE_WIDTH))
        cv2.putText(image,label,(50,50), font, 1,(255,0,0),2,cv2.LINE_AA)
        
        # Write image to video
        out.write(image)
        
        
        # Show the image
        cv2.imshow('image',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Release and destory everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
