import os
import numpy as np
import cv2
import cPickle as pickle
import gzip
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import log_loss

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
        self.cluster = MiniBatchKMeans(n_clusters=200)
        self.dir = 'Desktop/AI/Train'
        self.labelFiles = dict()
        for label in LABELS.keys():
            self.labelFiles[label] = [os.path.join(self.dir, label, ff) \
                                     for ff in filter(lambda fn: fn.endswith(".jpg"), \
                                        os.listdir(os.path.join(self.dir, label)))]


    def submission(self):
        testPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Test')
        subPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'submission.csv')
        train = []
        for label in LABELS.keys():
            clas = LABELS[label]
            train += [(fn, clas) for fn in self.labelFiles[label]]
        test = [os.path.join(testPath, ff) \
                for ff in filter(lambda fn: fn.endswith(".jpg"), os.listdir(testPath))]
        preds = self.predict(train, test)
        with open(subPath, 'wb') as subFile:
            subFile.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
            for i in range(len(test)):
                subFile.write(",".join([os.path.basename(test[i])] + [str(prob) for prob in preds[i]]) + "\n")

    def crossValidation(self, k):
        scores = []
        testsets = []
        for list in self.labelFiles.values():
            random.shuffle(list)
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
            index = 0
            for key in self.labelFiles.keys():
                for image in self.labelFiles[key]:
                    if index == k:
                        index = 0
                    if index == i:
                        test.append(image)
                        testLabels.append(LABELS[key])
                        
                    else:
                        train.append((image,LABELS[key]))

                    index += 1
            prediction = self.predict(train, test)
            scores.append(log_loss(testLabels,prediction))
            print (i+1) ," loops complete"
        print scores
        print "Mean Accuracy:", np.mean(scores)
        
        
    def getFeatures(self, train):
        features = []
        for fn in train:
            # Read in image, resize all images to same size (currently hardcoded)
            img = cv2.imread(fn)
            img_r = cv2.resize(img,(1280,720))
            # Only current feature: count of pixel intensities in grayscale image
            gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY) 
            # TODO: Add additional features

            img_blur = cv2.blur(img_r,(20,20))
            edges = cv2.Canny(img_blur,0,100)
            lapl = cv2.Laplacian(gray,cv2.CV_64F)
            b,g,r = cv2.split(img_r)
            surffeatures = self.surf(fn)
            values = self.histogram(surffeatures)
            features.append([np.sum(np.array(gray)),np.sum(np.array(edges)),np.sum(np.array(lapl)),np.sum(np.array(r)),np.sum(np.array(g)),np.sum(np.array(b))])
            features[-1]+= values
        return features

    #Makes the surf files of the test sets.   
    def preprocess(self,test):
        surfs = {}
        for image in test:
            des = self.surf(image)
            surfs[image] = des
            with gzip.open(image + ".zip",'wb') as f:
                pickle.dump(des,f)
    
    '''
    #gets the surf features of a single image. Has to be preprocessed
    def surf(self,image):
        features = [];
        with gzip.open(image + ".zip",'rb') as f:
            features = pickle.load(f)
        return features
    '''

    #gets the surf features of a single image after rescaling and grayscaling    
    def surf(self, image):
        surfClass = cv2.SURF()
        img = cv2.imread(image)
        img_r = cv2.resize(img,(1280,720))
        gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        kp,des = surfClass.detectAndCompute(gray,None)
        return des        
    
    #cluster for small amount of data    
    def make_cluster(self,data):
        all_surfs = self.surf(data[0][0])
        for image in data:
            if(image == data[0][0]):
                continue
            li = self.surf(image[0])
            toAdd = np.array(li)
            all_surfs = np.vstack((all_surfs,toAdd))
        self.cluster.fit(all_surfs)
    '''
    #cluster for with predone surf features    
    def make_cluster(self,data):
        self.cluster = MiniBatchKMeans(self.cluster.get_params()['n_clusters'])
        for i in data:
            i = self.surf(data[0])
            self.cluster.partial_fit(features)
    '''      
    def histogram(self,array):
        labels = self.cluster.predict(array)
        result = {}
        for i in range(0,self.cluster.get_params()['n_clusters']):
            result[i] = 0
        for label in labels:
            result[label]+=1
        for i in result.keys():
            result[i] = float(result[i])/(float(array.size)/float(128))*float(10000) #because i find the values too small
        arrayResult = []
        for i in result.values():
            arrayResult.append(i)
        return arrayResult
            
        
        
    def predict(self, train, test):
        self.make_cluster(train)
        print "Done with surfing and clustering"
        Train = [f for (f,i) in train]
        #noFishTrain = [f for f, l in filter(lambda (fn, lab) : lab == 0, train)]
        #fishTrain = [f for f, l in filter(lambda (fn, lab) : lab == 1, train)]
        #fishF = self.getFeatures(fishTrain)
        #noFishF = self.getFeatures(noFishTrain)
        testF = self.getFeatures(test)
        X = self.getFeatures(Train)
        y = [i for (f,i) in train]
        # TODO: Fit a classifier
        classifier = RandomForestClassifier(100)
        classifier.fit(X,y)
        prediction = classifier.predict_proba(testF)
        return prediction
        
        
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


    def scoreMulti(self, prediction, testLabels):
        counts = dict()
        corrects = dict()
        for lab in LABELS.keys():
            c = LABELS[lab]
            counts[c] = 0
            corrects[c] = 0
    
        for i in range(len(testLabels)):
            l = testLabels[i]
            p = prediction[i]
            counts[l] += 1
            if l == p:
                corrects[l] += 1
        return tuple([corrects[c]/float(counts[c]) for c in counts.keys()])


def match(pattern, tup):
    for (ix, val) in pattern:
        if tup[ix] != val: return False
    return True

def getAttrFromItemset(itemset, attrIx):
    for (ix, val) in itemset:
        if ix == attrIx: return val
    return -1

# Expects a matrix where every row is a transaction
# Returns a list of (itemset, support) pairs
# Itemset is a list of (attrIndex, value) pairs
def eclat(transactionTable, minsup):
    data = {}
    for rowIx in range(len(transactionTable)):
        row = transactionTable[rowIx]
        for item in enumerate(row):
            if not item in data:
                data[item] = set([rowIx])
            else:
                data[item].add(rowIx)
    return eclatImpl([], data.items(), minsup)

# http://adrem.ua.ac.be/~goethals/software/files/eclat.py
def eclatImpl(prefix, items, minsup):
    output = []
    while items:
        i,itids = items.pop()
        isupp = len(itids)
        if isupp >= minsup:
            output.append((sorted(prefix+[i]), isupp))
            suffix = [] 
            for j, ojtids in items:
                jtids = itids & ojtids
                if len(jtids) >= minsup:
                    suffix.append((j,jtids))
            output += eclatImpl(prefix+[i], sorted(suffix, key=lambda item: len(item[1]), reverse=True), minsup)
    return output

if __name__ == '__main__':
    fc = FishClassifier()
    fc.crossValidation(10)