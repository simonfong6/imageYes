import os
import cv2
import numpy as np
import json

class Dataset:
    def __init__(self,data_dir,height,width):
        self.data_dir = data_dir# Path to the directory of data
        self.height = height    # Height that an image is resized to
        self.width = width      # Width that an image is resized to
        self.channels = 3
        self.num_classes = 0    # Tracks number of classes
        self.image_count = 0    # Tracks total number of images
        self.image_paths = []   # List of paths to images
        self.images = []        # List of images
        self.names = {}         # Dictonary that maps vector labels to text 
        self.labels = []        # List of vector labels
        
        # Train fields
        self.num_train = 0
        self.image_paths_train = []
        self.images_train = []
        self.labels_train = []
        
        # Validation fields
        self.num_val = 0
        self.image_paths_val = []
        self.images_val = []
        self.labels_val = []
        
        # Test fields
        self.num_test = 0
        self.image_paths_test = []
        self.images_test = []
        self.labels_test = []
        
        
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
        
    def create_perm(self,length):
        """Returns a permutation for the given length"""
        perm = np.arange(length)
        return np.random.shuffle(perm)
        
    def shuffle(self,a_list,perm):
        """Returns a shuffled version of that list according to the perm"""
        
        return [ a_list[i] for i in perm ]
        
        
        
    def next_batch(self,batch_size):
        """Return the next `batch_size` examples from this data set."""
        
        # Create a random seed
        perm = self.create_perm(self.image_count)
        
        
        # Shuffle paths and get a batch of image arrays
        image_paths = self.shuffle(self.image_paths[],perm)
        image_paths = image_paths[:batch_size]
        batch_images = self.load_data(image_paths)
        
        # Shuffle labels and get a batch
        batch_labels = self.shuffle(self.labels[],perm)
        batch_labels = batch_labels[:batch_size]
        
        # Convert to numpy array
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)
        
        # Reshape to proper expected output
        batch_images.reshape(batch_size,self.height,self.width,self.channels)
        batch_labels.reshape(batch_size,self.num_classes)
        
        return batch_images, batch_labels
        
    def train_val_test_split(self,train_percent, val_percent, test_percent):
        """Splits data into training, validation, and test set."""
        
        # Check if valid values
        if((train_percent + val_percent + test_percent) != 100):
            print("Percents do not total to 100")
            return None
        
        # Calculate amount of each
        self.num_train = int( (train_percent/100.0) * self.image_count)
        self.num_val = int( (val_percent/100.0) * self.image_count)
        self.num_test = self.image_count - self.num_train - self.num_val
        
        
        
        
        
        
        
        
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
