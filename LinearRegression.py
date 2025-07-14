from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd


data = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
X = data[['sepal_length']]  # Feature(s)
y = data['petal_length']    # Target


X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)


model = LinearRegression()
model.fit(X_train, y_train)


preds = model.predict(X_val)


mae = mean_absolute_error(y_val, preds)
print("MAE:", mae)
