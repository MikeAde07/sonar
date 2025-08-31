#import the dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading the dataset to a pandas DataFrame
sonar_data = pd.read_csv('Sonar data.csv', header=None)

#separating data and labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

#instantiate model
model = LogisticRegression()

#split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify = Y, random_state=1)

#fit model to training data
model.fit(X_train, Y_train)

#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print("Accuracy on training data : ", training_data_accuracy)

#accuracy on test data
Y_pred = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_pred, Y_test)

print("Accuracy on testing data : ", test_data_accuracy)

# Making a predictive systems

input_data = (0.0129, 0.0141, 0.0309, 0.0375, 0.0767, 0.0787, 0.0662, 0.1108, 0.1777, 0.2245, 0.2431, 0.3134, 0.3206, 0.2917, 0.2249, 0.2347, 0.2143, 0.2939, 0.4898, 0.6127, 0.7531, 0.7718, 0.7432, 0.8673, 0.9308, 0.9836, 1, 0.9595, 0.8722, 0.6862, 0.4901, 0.328, 0.3115, 0.1969, 0.1019, 0.0317, 0.0756, 0.0907, 0.1066, 0.138, 0.0665, 0.1475, 0.247, 0.2788, 0.2709, 0.2283, 0.1818, 0.1185, 0.0546, 0.0219, 0.0204, 0.0124, 0.0093, 0.0072, 0.0019, 0.0027, 0.0054, 0.0017, 0.0024, 0.0029
)

# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the np array as we are predicting for one instance

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 'R'):
    print("The object is a rock")
else:
    print("The object is a Mine")