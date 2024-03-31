import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
    
def clearText(df):
    wordLemmatizer=WordNetLemmatizer()
    processedText=[]
    
    # Load stop words
    stopWords = set(stopwords.words('english'))
    mbtiTypes = list(np.unique(df.type))
    mbtiTypes = [p.lower() for p in mbtiTypes]
    
    print("Cleaning The Dataset")
    for sentence in tqdm(df.posts):
        
        sentence = sentence.lower()
        
        sentence = re.sub('https?://[^\s<>"]+|www\.[^\s<>"]+', ' ', sentence)
        
        sentence = re.sub('[^0-9a-z]', ' ', sentence)
        
        # Remove stop words
        sentence = " ".join([word for word in sentence.split() if word not in stopWords])
        
        # Remove potential labels
        for p in mbtiTypes:
            sentence = re.sub(p, '', sentence)
        
        # Lemmatize words
        sentence = wordLemmatizer.lemmatize(sentence) 
        
        processedText.append(sentence)

    return processedText  

def split(df, size):
    
    print("Splitting into train & test")
    trainRawData, testRawData = train_test_split(df, test_size=size, random_state=0, stratify=df.type)
    
    print("Applying Vectorization")
    vectorizer = TfidfVectorizer(max_features=12000)
    vectorizer.fit(trainRawData.posts)
    
    # vectorizer transform
    trainData = vectorizer.transform(trainRawData.posts).toarray()
    testData = vectorizer.transform(testRawData.posts).toarray()
    
    print("Label Encoding the classes")
    labelEncoder= LabelEncoder()
    
    print("Getting the final train and test")
    trainLabel = labelEncoder.fit_transform(trainRawData.type)
    testLabel = labelEncoder.fit_transform(testRawData.type)

    labelMapping = dict(zip(labelEncoder.classes_, labelEncoder.fit_transform(labelEncoder.classes_)))
    labelDict = dict([(value, key) for key, value in labelMapping.items()])

    return trainData, testData, trainLabel, testLabel, labelDict
