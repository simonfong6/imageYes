import os
import cv2
import numpy as np

class Data:
    def __init__(self,data_dir,height,width):
        self.data_dir = data_dir
        self.height = height
        self.width = width
        self.num_classes = 0
        self.images = []
        self.names = {}
        self.labels = []
        
    def load_data(self):
        dir_names = os.listdir(self.data_dir)
        num_classes = len(dir_names)
        for dir_name in dir_names:
            
            # Create a mapping from label number to name
            label = int(dir_name[:3])-1
            name = dir_name[3:]
            self.names[label] = name
            
            # Get images directory path
            dir_path = os.path.join(self.data_dir,dir_name)
            image_names = os.listdir(dir_path)
            
            for image_name in image_names:
                # Read, resize, and store image
                image_path = os.path.join(dir_path, image_name)
                image = cv2.imread(image_path)
                image_resized = cv2.resize(image, (self.height,self.width))
                self.images.append(image_resized)
                
                # Add image label array
                label_array = np.zeros((1,num_classes))
                print label
                print label_array.shape
                label_array[0,label] = 1
                self.labels.append(label_array)
    
                
        
def main():
    data = Data('256_ObjectCategories', 299, 299)
    data.load_data()
    print data.names
    
if __name__ == '__main__':
    main()
