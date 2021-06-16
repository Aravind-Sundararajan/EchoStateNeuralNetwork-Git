# General imports
import  pickle
import numpy as np
import scipy.io
from sklearn.preprocessing import OneHotEncoder

# Custom imports
from util.modules import RC_model

'''
this is a wrapper that does prediction for 1 trial use integer "type" to modify the model
'''
class rehappClassifier():
    #default constructor
    def __init__(self,type = 0):
        #setup config
        if type == 0:
            modelName = 'ESNN5np564Both042621'

        #unpickle the NN from file
        pickleFile = open('../output/'+modelName+'.txt', 'rb')
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
