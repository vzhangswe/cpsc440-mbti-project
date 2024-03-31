
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def model(xTrain, xTest, yTrain, yTest):
    modelsAccuracy={}
    
    #KNN
    print("Running KNN")
    kNeigh=KNeighborsClassifier(10)
    kNeigh.fit(xTrain, yTrain)
    modelsAccuracy['KNN']=accuracy_score(yTest, kNeigh.predict(xTest))
    
    #Logistic Regression
    print("Running Logistic Regression")
    logisticReg=LogisticRegression(max_iter = 10000, C = 2, n_jobs = -1)
    logisticReg.fit(xTrain,yTrain)
    modelsAccuracy['Logistic Regression']=accuracy_score(yTest, logisticReg.predict(xTest))

    #Linear SVC
    print("Running Linear SVC")
    linearSVC=LinearSVC(C = 0.5)
    linearSVC.fit(xTrain,yTrain)
    modelsAccuracy['Linear Support Vector Classifier']=accuracy_score(yTest,linearSVC.predict(xTest))

    #Multinomial Naive Bayes
    print("Running Multinomial Naive Bayes")
    multiNaiveBayes=MultinomialNB(alpha=5)
    multiNaiveBayes.fit(xTrain,yTrain)
    modelsAccuracy['Multinomial Naive Bayes']=accuracy_score(yTest,multiNaiveBayes.predict(xTest))
    
    #Random Forest
    print("Running Random Forest")
    model_forest=RandomForestClassifier(max_depth=50)
    model_forest.fit(xTrain,yTrain)
    modelsAccuracy['Random Forest Classifier']=accuracy_score(yTest,model_forest.predict(xTest))
    
    accuracyDF=pd.DataFrame(modelsAccuracy.items(),columns=['Models','Test accuracy'])

    return accuracyDF
    