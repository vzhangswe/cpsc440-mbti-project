import re
from tqdm import tqdm
import numpy as np
from keras.preprocessing.sequence import pad_sequences

labelToidx = {
    'INTJ': 0,
    'INTP': 1,
    'ENTJ': 2,
    'ENTP': 3,
    'INFJ': 4,
    'INFP': 5,
    'ENFJ': 6,
    'ENFP': 7,
    'ISTJ': 8,
    'ISFJ': 9,
    'ESTJ': 10,
    'ESFJ': 11,
    'ISTP': 12,
    'ISFP': 13,
    'ESTP': 14,
    'ESFP': 15
}

def cleanData(df):
    print("Cleaning The Dataset")
    maxLength = 0
    vocabSet = set()

    for i in tqdm(range(len(df.posts))):
        sentence = df.posts[i]
        sentence = sentence.lower()
        
        sentence = re.sub('https?://[^\s<>"]+|www\.[^\s<>"]+', '', sentence)
        sentence = sentence.replace("|||", " ")
        sentence = re.sub('[^0-9a-z]', ' ', sentence)

        if maxLength < len(sentence.split(' ')):
            maxLength = len(sentence.split(' '))
        vocabSet = vocabSet.union(set(sentence.split(' ')))
        df.posts[i] = sentence
    
    return df, vocabSet
        
def tokenizeData(trainData, testData, vocabSet, seqLength):
    wordToidx = {word: i for i, word in enumerate(vocabSet)}

    trainInput = [[wordToidx[word] for word in sentence.split(' ')] for sentence in trainData['posts']]
    trainInput = pad_sequences(maxlen=seqLength, sequences=trainInput, padding='post', value=0)

    testInput = [[wordToidx[word] for word in sentence.split(' ')] for sentence in testData['posts']]
    testInput = pad_sequences(maxlen=seqLength, sequences=testInput, padding='post', value=0)

    trainLabel = [[labelToidx[sentence]] for sentence in trainData['type']]
    trainLabel = np.array(trainLabel)

    testLabel = [[labelToidx[sentence]] for sentence in testData['type']]
    testLabel = np.array(testLabel)

    return trainInput, testInput, trainLabel, testLabel
