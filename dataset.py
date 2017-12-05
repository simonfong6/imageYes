import os
import cv2
import numpy as np
import json

class Dataset:
    def __init__(self,data_dir,height,width):
        self.data_dir = data_dir
        self.height = height
        self.width = width
        self.num_classes = 0
        self.image_count = 0
        self.image_paths = []
        self.images = []
        self.names = {}
        self.labels = []
        
    def read_data(self):
        """Creates a list of image paths and labels"""
        dir_names = os.listdir(self.data_dir)
        dir_names.sort()
        self.num_classes = len(dir_names)
        
        for index,dir_name in enumerate(dir_names):
            
            # Create a mapping from label number to name
            label = index
            name = dir_name
            self.names[label] = name
            
            # Get images directory path
            dir_path = os.path.join(self.data_dir,dir_name)
            image_names = os.listdir(dir_path)
            image_names.sort()
            
            for image_name in image_names:
                # Read, resize, and store image path
                image_path = os.path.join(dir_path, image_name)
                self.image_paths.append(image_path)
                
                # Add image label array
                # MAKE SURE THIS MATCHES WITH OTHER SIZES
                label_array = np.zeros((self.num_classes))
                label_array[label] = 1
                self.labels.append(label_array)
                # print "loaded {}".format(image_name)
                
                self.image_count += 1
        
        print("Loaded data from {}".format(self.data_dir))
        
    def load_data(self, image_paths):
        """Loads images as numpy arrays and returns them."""
        
        # Stores image arrays
        images = []
        
        # Read images and store in images list
        for image_path in image_paths:
            image = cv2.imread(image_path)
            image_resized = cv2.resize(image, (self.height,self.width))
            images.append(image_resized)
        
        return images
        
        
    def next_batch(self,batch_size):
        """Return the next `batch_size` examples from this data set."""
        
        # Create a random seed
        perm = np.arange(self.image_count)
        np.random.shuffle(perm)
        
        # Shuffle paths and get a batch of image arrays
        image_paths = [ self.image_paths[i] for i in perm ]
        image_paths = image_paths[:batch_size]
        batch_images = self.load_data(image_paths)
        
        # Shuffle labels and get a batch
        batch_labels = [ self.labels[i] for i in perm ]
        batch_labels = batch_labels[:batch_size]
        
        # Convert to numpy array
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)
        
        # Reshape to proper expected output
        batch_images.reshape(batch_size,self.height,self.width,3)
        batch_labels.reshape(batch_size,self.num_classes)
        
        return batch_images, batch_labels
    
                
        
def main():
    
    data = Dataset('train', 299, 299)
    data.read_data()
    
    #print json.dumps(data.names, indent=4, sort_keys=True)
    
    names = data.names
    
    images, labels = data.next_batch(50)

    for image,label in zip(images,labels):
        name = names[np.argmax(label)]
        print("{} \t {}".format(name,label))
        cv2.imshow(name,image)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
