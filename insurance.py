from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle

df = pd.read_csv('insurance\insurance3r2.csv')

dependent_variable=['insuranceclaim']
independent_variable=['age','sex','bmi','steps','children','smoker','region','charges']

x=df[independent_variable]
y=df[dependent_variable]



X_train, X_test, y_train, y_test = train_test_split( x,y , test_size = 0.2, random_state = 0)

clf = LogisticRegression(fit_intercept=True, max_iter=10000)
clf.fit(X_train, y_train)

pickle.dump(clf, open('insurance\model.pkl', 'wb'))