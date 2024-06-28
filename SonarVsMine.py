# Sonar VS Mine Project
## Using Superwise Learning Algorithm (Classification project)
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

sonar_data = pd.read_csv('sonar data.csv',header=None)
sonar_data.groupby(60).mean()

# Spliting data between lables and features
features = sonar_data.drop(columns=60,axis=1)   #axis =1 represent row
lables = sonar_data[60]

# Train Test Split
train_features, test_features, train_lables, test_lables = train_test_split(features,lables,test_size=0.1,stratify=lables,random_state=1)

# creating model
model = LogisticRegression()
#train the model on training data
model.fit(train_features,train_lables)

# testing model on small scale
some_features = test_features[0:10]
some_lables = test_lables[0:10]
some_predications = model.predict(some_features)
print(some_predications)
print(list(some_lables))

# Finding accuracy of model

# finding accuracy of train data
train_predications = model.predict(train_features)
train_pred_accuracy = accuracy_score(train_predications,train_lables)
print("Accuracy of train data predications is ",train_pred_accuracy)

# finding accuracy of test data
test_predictions = model.predict(test_features)
test_pred_accuracy = accuracy_score(test_predictions,test_lables)
print("Accuracy of test data predications is ",test_pred_accuracy)

# Predicating input data
input_data = [0.0762,0.0666,0.0481,0.0394,0.0590,0.0649,0.1209,0.2467,0.3564,0.4459,0.4152,0.3952,0.4256,0.4135,0.4528,0.5326,0.7306,0.6193,0.2032,0.4636,0.4148,0.4292,0.5730,0.5399,0.3161,0.2285,0.6995,1.0000,0.7262,0.4724,0.5103,0.5459,0.2881,0.0981,0.1951,0.4181,0.4604,0.3217,0.2828,0.2430,0.1979,0.2444,0.1847,0.0841,0.0692,0.0528,0.0357,0.0085,0.0230,0.0046,0.0156,0.0031,0.0054,0.0105,0.0110,0.0015,0.0072,0.0048,0.0107,0.0094]
type(input_data)
np_array = np.array(input_data)
np_array=np_array.reshape(1,-1)
input_data_predections = model.predict(np_array)
print(input_data_predections)

