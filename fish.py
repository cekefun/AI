import os
import numpy as np
import cv2
import gzip
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import pickle

# Fish Labels. NoF corresponds to No Fish, others are all types of Fish
LABELS = {
    "ALB" : 0,
    "BET" : 1,
    "DOL" : 2,
    "LAG" : 3,
    "NoF" : 4,
    "OTHER" : 5,
    "SHARK" : 6,
    "YFT" : 7,
}

class FishClassifier:
    def __init__(self):
        # Creates dictionary that maps every label to a list of files with that label
        self.index = 0
        self.dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Train')
        self.labelFiles = dict()
        for label in LABELS.keys():
            self.labelFiles[label] = [os.path.join(self.dir, label, ff) \
                                     for ff in filter(lambda fn: fn.endswith(".jpg"), \
                                        os.listdir(os.path.join(self.dir, label)))]


    def crossValidation(self, k):
        scores = []
        allItems = list(self.labelFiles.values())
        allItems = sum(allItems,[])
        indexStart = allItems.index(self.labelFiles["NoF"][0])
        indexEnd = allItems.index(self.labelFiles["NoF"][-1])
        for i in range(0, k):
            train = []
            test = []
            testLabels = []
            # TODO: Divide data in training and test set according to k
            # Make sure you maintain the original label distribution
            # train should be a list of (filename, class) pairstion
            # train should be a list of (filename, class) pairs
            # test should be a list of filenames
            # testLabels should be a list of classes in the same order as test
            # Class = 0 if NoFish, Class = 1 if some type of Fish

            for item in range(0,len(allItems)):
                picture = allItems[item]
                Class = 1
                if item >= indexStart and item <= indexEnd:
                    Class = 0

                if item % k == i:
                    test.append(picture)
                    testLabels.append(Class)
                else:
                    train.append((picture,Class))
            prediction = self.predict(train, test)
            scores.append(self.score(prediction, testLabels))
            print (i+1) ," loops complete"
        print scores
        print "Mean Accuracy:", np.mean([(x1+x2)/2.0 for (x1, x2) in scores])
        
        
    def getFeatures(self, train):
        features = []
        for fn in train:
            # Read in image, resize all images to same size (currently hardcoded)
            img = cv2.imread(fn)
            img_r = cv2.resize(img,(1280,720))
            # Only current feature: count of pixel intensities in grayscale image
            gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY) 
            # TODO: Add additional features
            features.append([np.sum(np.array(gray))])
            
            #img_blur = cv2.blur(img_r,(1,1))
            #edges = cv2.Canny(img_blur,100,200)
            #features.append([np.sum(np.array(edges))])

            #edges = cv2.Canny(img_blur,200,300)
            #features.append([np.sum(np.array(edges))])

            #edges = cv2.Canny(img_blur,0,100)
            #features.append([np.sum(np.array(edges))])
            
        return features
        
    def sift(self,train):
        sifts = {}
        siftClass = cv2.SIFT()
        for image in train:
            img = cv2.imread(image)
            kp, des = siftClass.detectAndCompute(img,None)
            sifts[image] = des
        with gzip.open("test"+str(self.index)+".zip", 'wb') as f:
            pickle.dump(sifts, f)

        self.index += 1
            
            
    def predict(self, train, test):
        self.sift(test)
        noFishTrain = [f for f, l in filter(lambda (fn, lab) : lab == 0, train)]
        fishTrain = [f for f, l in filter(lambda (fn, lab) : lab == 1, train)]
        fishF = self.getFeatures(fishTrain)
        noFishF = self.getFeatures(noFishTrain)
        testF = self.getFeatures(test)
        X = fishF + noFishF
        y = [1 for x in fishF] + [0 for x in noFishF]
        # TODO: Fit a classifier
        classifier = RandomForestClassifier()
        classifier.fit(X,y)
        return classifier.predict(testF)
        
        
    def score(self, prediction, testLabels):
        # Computes predictive accuracy for both classes;
        # Overall score is average of these 2 accuracies
        # This handles class imbalance on the scoring side
        zeroCount, zeroCorr, oneCount, oneCorr = 0, 0, 0, 0
        for i in range(len(testLabels)):
            if testLabels[i] == 0:
                if prediction[i] == 0:
                    zeroCorr += 1
                zeroCount += 1
            elif testLabels[i] == 1:
                if prediction[i] == 1:
                    oneCorr += 1
                oneCount += 1
        return (zeroCorr/float(zeroCount), oneCorr/float(oneCount))


if __name__ == '__main__':
    fc = FishClassifier()
    fc.crossValidation(10)