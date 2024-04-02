import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch import optim
from tqdm import tqdm
from transformer import Transformer
from transDataProcess import cleanData, tokenizeData

# parameters
maxSeqLength = 500 # in the training sample, 2584 is the most ideal one
embeddingDim = 64
numLabel = 16
epochs = 300 # Iteration Times
batchSize = 128
learningRate = 0.001 
device = 'cpu' 
filePath = './mbti_data.csv'

df = pd.read_csv(filePath)
dfClean, vocabSet = cleanData(df)
vocab_size = len(vocabSet)
trainRawData, testRawData = train_test_split(dfClean, test_size=0.2, random_state=0, stratify=df.type)

trainInput, testInput, trainLabel, testLabel = tokenizeData(trainRawData, testRawData, vocabSet, maxSeqLength)

# training data 
xTrain = torch.from_numpy(trainInput).to(torch.long)
yTrain = torch.from_numpy(trainLabel).to(torch.long)
trainDataset = TensorDataset(xTrain, yTrain)
trainLoader = torch.utils.data.DataLoader(trainDataset, batchSize, True)

# test data
testData = torch.from_numpy(testInput).to(torch.long)
testLabel = torch.from_numpy(testLabel).to(torch.long)
testDataSet = TensorDataset(testData, testLabel)
testLoader = torch.utils.data.DataLoader(testDataSet, batchSize, True)

model = Transformer(vocab_size, embeddingDim, numLabel)
optimizer = optim.Adam(model.parameters(), lr=learningRate)
criterion = nn.CrossEntropyLoss()

model.to(device)

accMax = 0
modelMax = None
earlyStop = 0

for epoch in range(epochs):
    model.train() # Train Mode
    epochAccCount = 0 
    epochTrainCount = 0
    
    for xTrain, yTrain in tqdm(trainLoader):

        xTrain = xTrain.to(device)
        optimizer.zero_grad()
        output = model(xTrain)
        
        # loss
        loss = criterion(output, yTrain.long().view(-1))
        loss.backward()
        optimizer.step()
        
        # Record train accuracy
        epochAccCount += (output.argmax(axis=1) == yTrain.view(-1)).sum()
        epochTrainCount += len(xTrain)
        
    # Calculate train accuracy
    epochTrainacc = epochAccCount / epochTrainCount
    
    print("EPOCH: %s" % str(epoch + 1))
    print("Accuracy: %s" % (str(epochTrainacc.item() * 100)[:5]) + '%')

    model.eval()
    accCount = 0
    with torch.no_grad():  # No need to track gradients during evaluation
        for inputs, targets in testLoader:
            output = model(inputs)
            accCount += (output.argmax(axis=1) == targets.view(-1)).sum()
        
        if epoch > 10 and (accCount - earlyStop) < -50: # Early Stop
            print("Stop at Test Accuracy %s" % (str(float(accCount / len(testLabel)) * 100)[:5]) + '%')
            print(f'Early Stop at {epoch}')
            break

        earlyStop = accCount
        accMax = epochTrainacc
        modelMax = model
        
        print("Test Accuracy %s" % (str(float(accCount / len(testLabel)) * 100)[:5]) + '%')
    
# save the best model
torch.save(modelMax, './bestModel.pkl')
print('Done')
