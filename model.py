from sklearn.svm import SVR
import pandas as pd

df = pd.read_csv(r'pong_data.csv')

X = df[df['ball_x']>650]['ball_y']
y = df[df['ball_x']>650]['paddle_y']

print(X)
print(y)

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=0)
Xtrain = Xtrain.values.reshape(-1, 1)
Xtest = Xtest.values.reshape(-1, 1)

model = SVR(kernel='linear')
model.fit(Xtrain, ytrain)

y_model = model.predict(Xtest)

from sklearn.metrics import r2_score
print(r2_score(Xtest, y_model))

from joblib import dump
dump(model, 'mymodel.joblib') #save 

print(model.predict([[50]]))