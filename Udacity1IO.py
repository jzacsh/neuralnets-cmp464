'''
Created on Jun 2, 2017

@author: bob
'''
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
#This is getting data for image recognition used in assignment 2
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from urllib.request import urlretrieve
import pickle

url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = '/Users/bob/Documents/TensorFlow/Data/' # Change me to store data elsewhere

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 5% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
        
def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  dest_filename = os.path.join(data_root, filename)
  if force or not os.path.exists(dest_filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(dest_filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', dest_filename)
  else:
    raise Exception(
      'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
  return dest_filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)
#Kluges not needed he always returns proper filenam
#train_filename = "/Users/bob/Documents/TensorFlow/Data/notMNIST_large.tar.gz" #Kluge so dont have to do again
#test_filename = "/Users/bob/Documents/TensorFlow/Data/notMNIST_small.tar.gz"

num_classes = 10
np.random.seed(133)

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall(data_root)
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]## this is called list comprehension with an if look at http://www.secnetix.de/olli/Python/list_comprehensions.hawk for instance
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders
  
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)
# Kluges below not needed he puts together even if not done again
#train_folders=['/Users/bob/Documents/TensorFlow/Data/notMNIST_large/A', '/Users/bob/Documents/TensorFlow/Data/notMNIST_large/B', '/Users/bob/Documents/TensorFlow/Data/notMNIST_large/C', '/Users/bob/Documents/TensorFlow/Data/notMNIST_large/D', '/Users/bob/Documents/TensorFlow/Data/notMNIST_large/E', '/Users/bob/Documents/TensorFlow/Data/notMNIST_large/F', '/Users/bob/Documents/TensorFlow/Data/notMNIST_large/G', '/Users/bob/Documents/TensorFlow/Data/notMNIST_large/H', '/Users/bob/Documents/TensorFlow/Data/notMNIST_large/I', '/Users/bob/Documents/TensorFlow/Data/notMNIST_large/J']
#test_folders = ['/Users/bob/Documents/TensorFlow/Data/notMNIST_small/A', '/Users/bob/Documents/TensorFlow/Data/notMNIST_small/B', '/Users/bob/Documents/TensorFlow/Data/notMNIST_small/C', '/Users/bob/Documents/TensorFlow/Data/notMNIST_small/D', '/Users/bob/Documents/TensorFlow/Data/notMNIST_small/E', '/Users/bob/Documents/TensorFlow/Data/notMNIST_small/F', '/Users/bob/Documents/TensorFlow/Data/notMNIST_small/G', '/Users/bob/Documents/TensorFlow/Data/notMNIST_small/H', '/Users/bob/Documents/TensorFlow/Data/notMNIST_small/I', '/Users/bob/Documents/TensorFlow/Data/notMNIST_small/J']
########################


#Pickling
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):#used by maybe_pickle look at first; folder is a letter folder
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)#returns a list of files in the letter directory
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)### an array of all the files in the directory;
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth #normalizing range to -1<x<=1; note imread makes an array
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data# actually putting images in an array -- note is so we have NX28X28 arrays to use in our neural net
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :] #because they have rejected some files
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):#making the A.pickle B.pickle etc from the A,B,... in the two  notMinist_large and notMNIst_small
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:##https://docs.python.org/2/tutorial/inputoutput.html#methods-of-file-objects  note wb means write binary
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

train_datasets = maybe_pickle(train_folders, 45000)# list of the names [/Users/bob/Documents/TensorFlow/Data/notMNIST_large/A.pickle ,...B.pickle ...] in nonMNIST.large
test_datasets = maybe_pickle(test_folders, 1800)#same idea
#print(train_datasets[1]);exit()
#exercise to display some of the images
#===============================================================================
# Abatch = pickle.load( open( train_datasets[0], "rb" ) )
# #print(Abatch)
# 
# a1 =plt.imshow(Abatch[3,:,:])
# plt.show()
# a2 =plt.imshow(Abatch[7,:,:])

# plt.show()
#===============================================================================
#exercise to compare shapes in all the train datasets
#===============================================================================
# for data_set in train_datasets:
#     data_array=  pickle.load( open( data_set, "rb" ) )
#     print(" the dataset %s  and shape %s " %(data_set, data_array.shape[0]))
#===============================================================================
#will stack them
#===============================================================================

#===============================================================================
def make_arrays(nb_rows, img_size):# just utility to make an array of right size
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):#pickle_files is the list of [directory/A.pickle,dir/B.pickle ...] in trainingset or testset; valid_size will be the validation size of 10000 for the training check
    #are going to shuffle the letters together but keep the labels correct as output
  num_classes = len(pickle_files)# this is 10 (a through J)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes  #// is whole quotient. Want equal numbers of each letter in trials
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):     #note enumerate give a touple for each element in a list so would get 0,pathtoA.pickle,1,pathtoB.pickle etc
    try:
      with open(pickle_file, 'rb') as f:# note is particular letter file 
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set  Note it shuffles on first axis which are the different symbols
        
        np.random.shuffle(letter_set) # mix up each of the  the As Bs etc; We will  have another shuffle to mix up the way presented below
        if valid_dataset is not None:# because for test data dont use a validation set
          valid_letter = letter_set[:vsize_per_class, :, :]# make an array of vsize_perclass with the 2d arrays of images
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label # putting all the As first then Bs note they all have same label.. Probably better to randomize the As and Bs
          start_v += vsize_per_class #going for next letter
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]  # same thing 
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels
            
            
train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)# very confusing as just saying give the one set for test with no validation 

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)
print(train_labels[0],"  ",train_labels[100], "  ",train_labels[19999],"  ", train_labels[20000])
# note labeled data is together 
#note large and small dat sets are for training vs testing not small and capital ascii_letters
#t there are groups of 20000 small letters together but mixed up each time called
# Note the indexing scheme we use to make sure its the same permutation for the dataset and the labels
def randomize(dataset, labels): # ths is to mix them all up but keeping labels with the correct data. We make a permutation of index into the array of data
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
print(train_labels[0],"  ",train_labels[1], "  ",train_labels[19999],"  ", train_labels[20000])
#note has been shuffled


####now to save in compressed pickle file
pickle_file = os.path.join(data_root, 'notMNIST.pickle') # show in file

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)#we will now be able to load the dictionary back in and are up and running to get training_dataset etc.
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
