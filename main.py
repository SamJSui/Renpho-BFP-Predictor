import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

dataframe = pd.read_csv('data.csv')
dataframe.isnull().values.any()

df = dataframe.drop(columns=['Remarks', 'Time of Measurement'])
df.drop(df.index[df['Body Fat(%)'] == '--'], inplace=True)

X = np.array(df['Body Fat(%)'])
X = X.reshape(-1,1)
Y = df.drop(columns=['Body Fat(%)']).to_numpy()
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=2)
model = LinearRegression()
model.fit(Xtrain, Ytrain)
print("MODEL IS READY")

while True:
    index = 0
    bodyFat = float(input())
    pred = np.array(bodyFat)
    pred = pred.reshape(-1, 1)
    prediction = model.predict(pred)
    # print("Weight:", prediction[0][0])
    # print("BMI:", prediction[0][1])
    # print("Fat-free Body Weight(lb):", prediction[0][1])
    # print("Subcutaneous Fat(%):", prediction[0][1])
    # print("Visceral Fat:", prediction[0][1])
    # print("Body Water(%):", prediction[0][1])
    # print("Visceral Fat:", prediction[0][1])
    for columnName in df.drop(columns=['Body Fat(%)']).columns:
        print("%s: %.2f" % (columnName, prediction[0][index]))
        index += 1

