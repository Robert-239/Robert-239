'''Module containing various utility functions

@since: 10 Jan 2012

@author: skroon, bherbst
'''
from warnings import warn

import numpy as np
import matplotlib.pyplot as plt


###############################################################
def nearest_feature(data,means):
    '''
    Classify data values to the class represented by the nearest feature vector.
    Each feature vector is assigned to one of k classes. Class j is represented by 
    means[:,j], j = 0,...,k-1
    
    Parameters
    ----------
    data : (n,d) ndarray 
         n features each of dimension d. 
    means : (k,d) ndarray
         The arrays representing the k different classes.
         
    Return
    ------
    labels : (n,) int array
           The class labels for each feature. 
           
    Note 
    ----
    This function uses broadcasting to efficiently do the assignments.
    
    Example
    -------
    >>>from utils import nearest_feature as nf
    >>>data = np.array([[0.5,0.5],[1.5,2.9],[2.1,3.9],[0.1,-0.1]])
    >>>data = np.array([[0.5,1.5,2.1,0.1],[0.5,2.9,3.9,-0.1]])
    >>>means = np.array([[0,0],[1,3],[2,4]])
    >>>means = np.array([[0,1,2],[0,3,4]])
    >>>labels = nf(data,means)
    >>>print (labels)
    Out:
       [0 1 2 0]
    '''
    dist = data[:,np.newaxis,:]-means
    
    # Insert one line of code
    
    err = np.linalg.norm(dist,axis=2)

    labels = np.argmin(err,axis=1)
    return labels

    
def confusion(orig,pred):
    '''
    Generate and print a confusion matrix. This version works for 
    Python 3.5 
    
    For the printing, the column widths containing
    the numbers should all be equal, and should be wide enough to accommodate the widest class name as
    well as the widest value in the matrix.
    
    Parameters
    ----------
    truth : (n,) list
        A list of the true class label for each data value.
        There are n data values.
    pred  : (n,) list
        A list of the class labels as returned by the system.
        
    Return
    ------
    result : dict
        A dictionary of the confusion matrix.
        
    Example
    -------

    >>> orig = ["Yellow", "Yellow", "Green", "Green", "Blue", "Yellow"]
    >>> pred = ["Yellow", "Green", "Green", "Blue", "Blue", "Yellow"]
    >>> result = confusion(orig, pred)
             Blue  Green Yellow
      Blue      1      0      0
     Green      1      1      0
    Yellow      0      1      2
    >>> result
    {('Yellow', 'Green'): 1, ('Green', 'Blue'): 1, ('Green', 'Green'): 1, ('Blue', 'Blue'): 1, ('Yellow', 'Yellow'): 2}
    '''
    print_  = True
    classes = set(orig)
    classes = classes.union(set(pred))
    classes = list(classes)
    conf = {}
    for i, c in enumerate(orig):
        if conf.get((c, pred[i]), 0):
            conf[c, pred[i]] += 1
        else:
            conf[c, pred[i]] = 1

    if print_:
        max_ = 0
        for c in classes:
            if len(str(c)) > max_:
                max_ = len(str(c))
        for c in classes:
            for d in classes:
                if len(str(conf.get((c, d), 0))) > max_:
                    max_ = len(str(conf.get((c, d), 0)))
    
        print (" ".rjust(max_),end="   ")
        for c in classes:       
            print ( str(c).rjust(max_),end=" ")
        print ()
        for c in classes:
            print (str(c).rjust(max_)," ",end=" ")
            for d in classes:
                print(str(conf.get((c,d),0)).rjust(max_),end=" ")
            print()
    return conf
    
    

def plot_confusion_matrix(cm, title='Confusion matrix', target_names = np.array(['setosa', 'versicolor', 'virginica'], 
      dtype='<U10'), cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def loadimages():
    import matplotlib.pyplot as plt
    import os
    import fnmatch
    """
    Load all the gray scale images in all the subdirectories with suffix `png`.
    The images are flattened and each image is represented as an (d,) array.
    
    Return
    ------
    
    images : (d,n) ndarray
       returns n, d-dimensional images.
    
    """
    matches = []
    for root, dirs, files in os.walk("./data/faces"):
        for filename in fnmatch.filter(files, '*.png'):
            matches.append(os.path.join(root, filename))
    data = []
    for m in matches:
        data.append(plt.imread(m).flatten())
    return np.column_stack(data)

def read_files_in_directory(dir_path):
    """
    Read diferent files from a directory.
    The path to the directory relative to current directory.
    This is a snippet that should be adapted for use in your 
    code
    
    Parameters
    ----------
    
    dir_path : char
       The directory containing the files
       
    Output
    ------
    
    In this snippet all files will be copied to to *.out
    
    Example
    -------
    read_files_in_directory('./data/sign/sign1/*.txt')
    """
    import glob
    list_of_files = glob.glob(dir_path)           # create the list of file
    for file_name in list_of_files:
       FI = open(file_name, 'r')
       FO = open(file_name.replace('txt', 'out'), 'w') 
    for line in FI:
       FO.write(line)

    FI.close()
    FO.close()
    
def read_images():
    """
    Use the skimage to read multiple images from a file.
    Reads all the png files from  all the directories in the current directory.
    
    Return
    ------
    data : (d,n) nd ndarray
       d is the dimension of the flattened images
       n is the number of images
    """
    from skimage import io
    import numpy as np

    ic = io.ImageCollection('*/*.png')
    data = np.array(ic)
    return data.reshape((len(data), -1))


