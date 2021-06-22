# General imports
import os
import  pickle
import numpy as np
import scipy.io
from sklearn.preprocessing import OneHotEncoder

# Custom imports
from echostate.model.mod import RC_model
'''
this is a wrapper that does prediction for 1 trial use integer "type" to modify the model
'''
class rehappClassifier():
    #default constructor
    def __init__(self,fname):
        #setup config
        #unpickle the NN from file
        pickleFile = open(fname, 'rb')
        self.classifier = pickle.load(pickleFile)
        pickleFile.close()

    #repr
    def __repr__(self):
        return "rehappClassifier{\n" + str(self.classifier) + "\n}"

    #returns the prediction class for an input trial file
    def predict(self, fname):
        test = np.genfromtxt(fname, delimiter=",")
        Xte = np.dstack(test.T)
        return self.classifier.predictions(Xte)
