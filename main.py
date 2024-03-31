import pandas as pd
import numpy as np

from splitVectorizeData import clearText, split
from model import model

df = pd.read_csv('./mbti_data.csv', index_col=False)
dfClean = df
dfClean.posts = clearText(df)

XTrain, XTest, yTrain, yTest, labelDict = split(dfClean, 0.2)

accuracyDF = model(XTrain, XTest, yTrain, yTest)
print(accuracyDF)





