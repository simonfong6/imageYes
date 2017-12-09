import os
import cv2          # Needed image loading
import numpy as np
import sys          # Needed for progress tracking

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
        
        # Calculate total for progress tracker
        total = len(image_paths)
        
        # Stores image arrays
        images = []
        
        # Read images and store in images list
        for i,image_path in enumerate(image_paths):
            image = cv2.imread(image_path)
            image_resized = cv2.resize(image, (self.height,self.width))
            images.append(image_resized)
            
            # Update progress tracker
            sys.stdout.write("Loading progress:    %d/%d   \r" % (i+1,total) )
            sys.stdout.flush()
        
        return images
        
    def create_perm(self,length,seed=0):
        """Returns a permutation for the given length"""
        
        np.random.seed(seed)
        perm = np.arange(length)
        np.random.shuffle(perm)

        return perm
        
    def shuffle(self,a_list,perm):
        """Returns a shuffled version of that list according to the perm"""
        
        return [ a_list[i] for i in perm ]
        
    def reshape_images(self, images):
        """Reshapes images to numpy arrays"""
        
        length = len(images)
        images = np.array(images)
        images.reshape(length,self.height,self.width,self.channels)
        
        return images
    
    def reshape_labels(self, labels):
        """Reshapes labels to numpy arrays"""
        
        length = len(labels)
        labels = np.array(labels)
        labels.reshape(length,self.num_classes)
        
        return labels
        
        
    def next_batch(self,batch_size):
        """Return the next `batch_size` examples from this data set."""
        
        # Create a random seed
        perm = self.create_perm(self.image_count)
        
        # Shuffle paths and get a batch of image arrays
        image_paths = self.shuffle(self.image_paths,perm)
        image_paths = image_paths[:batch_size]
        batch_images = self.load_data(image_paths)
        
        # Shuffle labels and get a batch
        batch_labels = self.shuffle(self.labels,perm)
        batch_labels = batch_labels[:batch_size]
        
        # Convert to numpy arrays
        batch_images = self.reshape_images(batch_images)
        batch_labels = self.reshape_labels(batch_labels)
        
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
        
        # Create a random seed
        perm = self.create_perm(self.image_count)
        
        # Shuffle
        image_paths = self.shuffle(self.image_paths,perm)
        
        # Shuffle labels
        labels = self.shuffle(self.labels,perm)
        
        # Assign training set
        start = 0
        end = self.num_train
        self.image_paths_train = image_paths[start:end]
        self.labels_train = labels[start:end]
        
        # Assign validaiton set
        start = end
        end = start + self.num_val
        self.image_paths_val = image_paths[start:end]
        self.labels_val = labels[start:end]
        
        # Assign test set
        start = end
        end = start + self.num_test
        self.image_paths_test = image_paths[start:end]
        self.labels_test = labels[start:end]
        
        print("Splits training, validation, and testing data.")
        
    def load_training(self):
        """Returns training data and labels"""
        
        # Load and reshape images
        print("Loading training data...")
        images_train = self.load_data(self.image_paths_train)
        self.images_train = self.reshape_images(images_train)
        
        # Reshape labels
        self.labels_train = self.reshape_labels(self.labels_train)
        
        print("Loaded training data")
        
        
        return self.images_train, self.labels_train
        
    def load_validation(self):
        """Returns validation data and labels"""
        
        # Load and reshape images
        print("Loading validation data...")
        images_val = self.load_data(self.image_paths_val)
        self.images_val = self.reshape_images(images_val)
        
        # Reshape labels
        self.labels_val = self.reshape_labels(self.labels_val)
        
        print("Loaded validation data")
        
        return self.images_val, self.labels_val
    
    def load_test(self):
        """Returns testing data and labels"""
        
        # Load and reshape images
        print("Loading testing data...")
        images_test = self.load_data(self.image_paths_test)
        self.images_test = self.reshape_images(images_test)
        
        # Reshape labels
        self.labels_test = self.reshape_labels(self.labels_test)
        
        print("Loaded validation data")
        
        return self.images_test, self.labels_test
        
        
        
        
        
def main():
    
    data = Dataset('caltech', 299, 299)
    data.read_data()
    data.train_val_test_split(1,39,60)
    
    names = data.names
    
    images, labels = data.load_training()

    # Display image and its label
    for i,image,label in zip(range(10),images,labels):
        name = names[np.argmax(label)]
        print("{} \t {}".format(name,np.argmax(label)))
        cv2.imshow(name,image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        if i == 10:
            break
    
if __name__ == '__main__':
    main()
