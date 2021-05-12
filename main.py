import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

data = pd.read_csv('data.csv', sep=',')
data = data[['date', 'deaths']]
# print(data.head())

x = np.array(data['date']).reshape(-1,1)
y = np.array(data['deaths']).reshape(-1,1)

plt.plot(y, '-g')

poly_fea = PolynomialFeatures(degree=4)
x = poly_fea.fit_transform(x)

model = linear_model.LinearRegression()
model.fit(x,y)

accuracy = model.score(x,y)
h = round(accuracy*100, 3)
print("\n" + "Network Accuracy {}".format(h))

y_ = model.predict(x)
plt.plot(y_, '--b')

days = 142
l = round(int(model.predict(poly_fea.fit_transform([[131+days]]))),2)
print("\n" + "Predicted Death Count(Total) is {}".format(l))

alpha = np.array(list(range(1, 131+days))).reshape(-1,1)
beta = model.predict(poly_fea.fit_transform(alpha))
plt.plot(beta, '--r')
plt.show()
