import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv('linear_regression_marks_hours.csv')
df = pd.DataFrame(data)
#selecting the column data
X = df[['Hours','Marks']]
print(X)
train= KMeans(df)
model = KMeans(n_clusters=2, random_state=0)
model.fit(X)
df['cluster'] = model.labels_
plt.scatter(df['Hours'], df['Marks'], c=df['cluster'])
plt.xlabel('Hours') 
plt.ylabel('Marks')
plt.title('KMeans Clustering of Hours and Marks')
plt.show()
