# -*- encoding: utf-8 -*-

import os
from time import time
import argparse
import scipy
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from sklearn.externals import joblib
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np 


# Author:
# Carlos Méndez-Cruz

# Goal: training, crossvalidation and testing transcription factor structural domain sentences

# Parameters:
# 1) --inputPath Path to read input files.
# 2) --inputTrainingData File to read training data.
# 4) --inputTrainingClasses File to read training classes.
# 3) --inputTestingData File to read testing data.
# 4) --inputTestingClasses File to read testing classes.
# 5) --outputModelPath Path to place output model.
# 6) --outputModelFile File to place output model.
# 7) --outputReportPath Path to place evaluation report.
# 8) --outputReportFile File to place evaluation report.
# 9) --classifier Classifier: BernoulliNB, SVM, kNN.
# 10) --saveData Save matrices
# 11) --kernel Kernel
# 12) --reduction Feature selection or dimensionality reduction
# 13) --removeStopWords Remove most frequent words
# 14) --ngrinitial N-gram inicial
# 15) --ngrfinal N-gram final
# 14) --vectorizer Vectorizer: b=binary, f=frequency, t=tf-idf.


# Ouput:
# 1) Classification model and evaluation report.

# Execution:

# source activate python3
# python training-crossvalidation-testing-dom-v1.py
# --inputPath /home/text-dom-dataset
# --inputTrainingData train-data.txt
# --inputTrainingClasses train-classes.txt
# --inputTestingData test-data.txt
# --inputTestingClasses test-classes.txt
# --outputModelPath /home/text-dom-dataset/models
# --outputModelFile SVM-lineal-model.mod
# --outputReportPath /home/text-dom-dataset/reports
# --outputReportFile SVM-linear.txt
# --classifier SVM
# --saveData
# --kernel linear
# --vectorizer b
# --ngrinitial 1
# --ngrfinal 1

# python3 training-crossvalidation-testing-dom-v1.py --inputPath /Users/elisulvaran/Desktop/LCG/Cuarto_Semestre/Ciencia_de_datos/Carlos_Mendez/Proyecto/binding-thrombin/ --inputTrainingData training_data.txt --inputTrainingClasses training_classes.txt --inputTestingData testing_data.txt --inputTestingClasses testing_classes.txt --outputModelPath /Users/elisulvaran/Desktop/LCG/Cuarto_Semestre/Ciencia_de_datos/Carlos_Mendez/Proyecto/binding-thrombin/models --outputModelFile MNB-model.mod --outputReportPath /Users/elisulvaran/Desktop/LCG/Cuarto_Semestre/Ciencia_de_datos/Carlos_Mendez/Proyecto/binding-thrombin/reports --outputReportFile MNB1.txt --classifier MultinomialNB --selection CHI2 --reduction PCA --numberElementsSelection 1000 --numberElementsReduction 24

###########################################################
#                       MAIN PROGRAM                      #
###########################################################

if __name__ == "__main__":
    # Parameter definition
    parser = argparse.ArgumentParser(description='training, crossvalidation and testing transcription factor structural domain sentences.')
    parser.add_argument("--inputPath", dest="inputPath",
                      help="Path to read input files", metavar="PATH")
    parser.add_argument("--inputTrainingData", dest="inputTrainingData",
                      help="File to read training data", metavar="FILE")
    parser.add_argument("--inputTrainingClasses", dest="inputTrainingClasses",
                      help="File to read training classes", metavar="FILE")
    parser.add_argument("--inputTestingData", dest="inputTestingData",
                      help="File to read testing data", metavar="FILE")
    parser.add_argument("--inputTestingClasses", dest="inputTestingClasses",
                      help="File to read testing classes", metavar="FILE")
    parser.add_argument("--outputModelPath", dest="outputModelPath",
                      help="Path to place output model", metavar="PATH")
    parser.add_argument("--outputModelFile", dest="outputModelFile",
                      help="File to place output model", metavar="FILE")
    parser.add_argument("--outputReportPath", dest="outputReportPath",
                      help="Path to place evaluation report", metavar="PATH")
    parser.add_argument("--outputReportFile", dest="outputReportFile",
                      help="File to place evaluation report", metavar="FILE")
    parser.add_argument("--classifier", dest="classifier",
                      help="Classifier", metavar="NAME",
                      choices=('BernoulliNB', 'SVM', 'MultinomialNB'), default='SVM')
    parser.add_argument("--saveData", dest="saveData", action='store_true',
                      help="Save matrices")
    parser.add_argument("--kernel", dest="kernel",
                      help="Kernel SVM", metavar="NAME",
                      choices=('linear', 'rbf', 'poly'), default='linear')
    parser.add_argument("--selection", dest="selection",
                      help="Feature selection", metavar="NAME",
                      choices=('CHI2', 'MI', 'PCA', 'SVD', 'tSNE'), default=None)
    parser.add_argument("--numberElementsSelection", type=int,
                      dest="num_elements_selection", required=True,
                      help="Number of elements for feature selection", metavar="INTEGER")
    parser.add_argument("--perplexity", type=int, 
                        dest="perplexity", default=30, required=False,
                        help="Perplexity of t-SNE. It is only necesary if you use t-SNE", metavar="INTEGER")

    parser.add_argument("--method-positive", metavar='NAME',
                        dest="met_int", default=None, required=False, choices=('squares','absolute'),
                        help="Method to make reduced training values positive.")


    args = parser.parse_args()

    # Printing parameter values
    print('-------------------------------- PARAMETERS --------------------------------')
    print("Path to read input files: " + str(args.inputPath))
    print("File to read training data: " + str(args.inputTrainingData))
    print("File to read training classes: " + str(args.inputTrainingClasses))
    print("File to read testing data: " + str(args.inputTestingData))
    print("File to read testing classes: " + str(args.inputTestingClasses))
    print("Path to place output model: " + str(args.outputModelPath))
    print("File to place output model: " + str(args.outputModelFile))
    print("Path to place evaluation report: " + str(args.outputReportPath))
    print("File to place evaluation report: " + str(args.outputReportFile))
    print("Classifier: " + str(args.classifier))
    print("Save matrices: " + str(args.saveData))
    if(args.classifier=='SVM'):
        print("Kernel: " + str(args.kernel))
    else:
        print('Positive values method: ' + str(args.met_int))
    if args.selection == 'MI' or args.selection == 'CHI2':
        print("Feature selection: " + str(args.selection))
    else:
        if args.selection is not None:
            print('Dimensionality reduction: ' + str(args.selection))
    if(args.selection=='tSNE'):
        print('Perplexity: ' + str(args.perplexity))


    # Start time
    t0 = time()


    y_train = []
    trainingData = []
    y_test = []
    testingData = []
    X_train = None
    X_test = None

    if args.saveData:
        print("Reading training data and true classes...")
        with open(os.path.join(args.inputPath, args.inputTrainingClasses), encoding='utf8', mode='r') \
                as iFile:
            for line in iFile:
                line = line.strip('\r\n')
                y_train.append(line)
        with open(os.path.join(args.inputPath, args.inputTrainingData), encoding='utf8', mode='r') \
                as iFile:
            for line in iFile:
                line = line.strip('\r\n')
                line = line.split(',')
                trainingData.append(line)
        print("   Done!")

        print("Reading testing data and true classes...")
        with open(os.path.join(args.inputPath, args.inputTestingClasses), encoding='utf8', mode='r') \
                as iFile:
            for line in iFile:
                line = line.strip('\r\n')
                y_test.append(line)
        with open(os.path.join(args.inputPath, args.inputTestingData), encoding='utf8', mode='r') \
                as iFile:
            for line in iFile:
                line = line.strip('\r\n')
                line = line.split(',')
                testingData.append(line)

        X_test = csr_matrix(testingData, dtype='double')
        X_train = csr_matrix(trainingData, dtype='double')
        print("   Done!")


        print("   Saving matrix and classes...")
        joblib.dump(X_train, os.path.join(args.outputModelPath, args.inputTrainingData + '.jlb'))
        joblib.dump(y_train, os.path.join(args.outputModelPath, args.inputTrainingData + '.class.jlb'))
        joblib.dump(X_test, os.path.join(args.outputModelPath, args.inputTestingData + '.jlb'))
        joblib.dump(y_test, os.path.join(args.outputModelPath, args.inputTestingClasses + '.class.jlb'))
        print("      Done!")
    else:
        print("   Loading matrix and classes...")
        X_train = joblib.load(os.path.join(args.outputModelPath, args.inputTrainingData + '.jlb'))
        y_train = joblib.load(os.path.join(args.outputModelPath, args.inputTrainingData + '.class.jlb'))
        X_test = joblib.load(os.path.join(args.outputModelPath, args.inputTestingData + '.jlb'))
        y_test = joblib.load(os.path.join(args.outputModelPath, args.inputTestingClasses + '.class.jlb'))
        print("      Done!")

    print("   Number of training classes: {}".format(len(y_train)))
    print("   Number of training class A: {}".format(y_train.count('A')))
    print("   Number of training class I: {}".format(y_train.count('I')))
    print("   Shape of training matrix: {}".format(X_train.shape))

    print("   Number of testing classes: {}".format(len(y_test)))
    print("   Number of testing class A: {}".format(y_test.count('A')))
    print("   Number of testing class I: {}".format(y_test.count('I')))
    print("   Shape of testing matrix: {}".format(X_test.shape))



    # Feature selection and dimensional reduction
    if args.selection is not None:
        num_elements_selection=int(args.num_elements_selection)

        if args.selection == 'CHI2':
            print('Performing feature selection...', args.selection)
            selec = SelectKBest(chi2, k=num_elements_selection)
            X_train = selec.fit_transform(X_train, y_train)

        elif args.selection == 'MI':
            print('Performing feature selection...', args.selection)
            selec = SelectKBest(mutual_info_classif, k=num_elements_selection)
            X_train = selec.fit_transform(X_train, y_train)

        elif args.selection == 'SVD':
            print('Performing dimensionality reduction...', args.selection)
            selec = TruncatedSVD(n_components=num_elements_selection, random_state=42)
            X_train = selec.fit_transform(X_train)

        elif args.selection == 'tSNE':
            print('Performing dimensionality reduction...', args.selection)
            perp=int(args.perplexity)
            selec = TSNE(n_components=num_elements_selection, perplexity=perp, random_state=42)
            X_train = selec.fit_transform(X_train.toarray(), y_train)

        elif args.selection == 'PCA':
            print('Performing dimensionality reduction...', args.selection)
            selec = PCA(n_components=num_elements_selection)
            X_train = selec.fit_transform(X_train.toarray(), y_train)


        print("   Done!")
        print('     New shape of training matrix: ', X_train.shape)

    
if args.classifier == 'MultinomialNB' or args.classifier == 'BernoulliNB':
        if args.selection == 'MI' or args.selection == 'CHI2':
            pass
        else:
            if args.selection is not None:
                if args.met_int == 'absolute':
                    X_train = np.absolute(X_train)
                elif args.met_int == 'squares':
                    X_train = np.square(X_train)



    jobs = 2
    paramGrid = []
    nIter = 300
    crossV = 3
    print("Defining randomized grid search...")
    if args.classifier == 'SVM':
        # SVM
        classifier = SVC()
        if args.kernel == 'rbf':
            paramGrid = {'C': scipy.stats.expon(scale=100),
                         # 'gamma': scipy.stats.expon(scale=.1),
                         'kernel': ['rbf'],
                         'class_weight': ['balanced', None]}
        elif args.kernel == 'linear':
            paramGrid = {'C': scipy.stats.expon(scale=100),
                         'kernel': ['linear'],
                         'class_weight': ['balanced', None]}
        elif args.kernel == 'poly':
            paramGrid = {'C': scipy.stats.expon(scale=100),
                         # 'gamma': scipy.stats.expon(scale=.1),
                         'degree': [2, 3],
                         'kernel': ['poly'],
                         'class_weight': ['balanced', None]}
        myClassifier = model_selection.RandomizedSearchCV(classifier,
                    paramGrid, n_iter=nIter,
                    cv=crossV, n_jobs=jobs, verbose=3)
    elif args.classifier == 'BernoulliNB':
        # BernoulliNB
        classifier = BernoulliNB()
        paramGrid = {'alpha': scipy.stats.expon(scale=1.0)}
        myClassifier = model_selection.RandomizedSearchCV(classifier, paramGrid, n_iter=nIter, cv=crossV, n_jobs=jobs, verbose=3)
    elif args.classifier == 'MultinomialNB':
        # MultinomialNB
        classifier = MultinomialNB()
        paramGrid = {'alpha': scipy.stats.expon(scale=1.0)}
        myClassifier = model_selection.RandomizedSearchCV(classifier, paramGrid, n_iter=nIter, cv=crossV, n_jobs=jobs, verbose=3)
    else:
        print("Bad classifier")
        exit()
    print("   Done!")

    print("Training...")
    myClassifier.fit(X_train, y_train)
    print("   Done!")


    print("Testing (prediction in new data)...")


    print('Shape of matrix before reduction: ', X_test.shape)




    if args.selection is not None:
        if args.selection == 'PCA' or args.selection == 'tSNE':
            X_test = selec.transform(X_test.toarray())
        else:
            X_test = selec.transform(X_test)



    print('Shape of matrix after reduction: ', X_test.shape)




    y_pred = myClassifier.predict(X_test)
    best_parameters = myClassifier.best_estimator_.get_params()
    if args.classifier == "SVM":
        confidence_scores = myClassifier.decision_function(X_test)
    print("   Done!")



    print("Saving report...")
    with open(os.path.join(args.outputReportPath, args.outputReportFile), mode='w', encoding='utf8') as oFile:
        oFile.write('**********        EVALUATION REPORT     **********\n')
        oFile.write('Classifier: {}\n'.format(args.classifier))

        if(args.selection == 'MI' or args.selection == 'CHI2'):
            oFile.write('Feature selection: {}, with k={}.\n'.format(args.selection,num_elements_selection))

        else:
            if args.selection is not None:
                oFile.write('Dimensionality reduction: {}, with {} principal components.\n'.format(args.selection,num_elements_selection))
        oFile.write('Method to make training data positive: {}'.format(args.met_int))
        oFile.write('\n')
        oFile.write('Training best score : {}\n'.format(myClassifier.best_score_))
        oFile.write('Accuracy: {}\n'.format(accuracy_score(y_test, y_pred)))
        oFile.write('Precision: {}\n'.format(precision_score(y_test, y_pred, average='weighted')))
        oFile.write('Recall: {}\n'.format(recall_score(y_test, y_pred, average='weighted')))
        oFile.write('F-score: {}\n'.format(f1_score(y_test, y_pred, average='weighted')))
        oFile.write('\n')
        oFile.write('Confusion matrix: \n')
        oFile.write(str(confusion_matrix(y_test, y_pred)) + '\n')
        oFile.write('\n')
        oFile.write('Classification report: \n')
        oFile.write(classification_report(y_test, y_pred) + '\n')
        oFile.write('Best parameters: \n')
        for param in sorted(best_parameters.keys()):
            oFile.write("\t%s: %r\n" % (param, best_parameters[param]))
        if args.classifier == "SVM":
            oFile.write('Number of support vectors per class: \n{}\n'.format(myClassifier.best_estimator_.n_support_))
            oFile.write('Support vectors: \n{}\n'.format(myClassifier.best_estimator_.support_vectors_))

    print("   Done!")

    print("Training and testing done in: %f s" % (time() - t0))