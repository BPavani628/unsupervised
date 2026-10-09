import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
## Load the dataset
data = {'age':[18,20,21,42,44,47],'spending':[20,21,22,40,44,45]}
df = pd.DataFrame(data)
print(df)
#selecting the column data
X = df[['age','spending']]
print(X)
#using kmeans to cluster the data
train= KMeans(df)

#this model is used to group the data into 2 clusters based on the features provided
#the kmeans algorithm will partition the data into a specified number of cluster
#minimizing the variance within each cluster
model = KMeans(n_clusters=2, random_state=0)
model.fit(X)

df['cluster'] = model.labels_
print(df)

plt.scatter(df['age'], df['spending'], c=df['cluster'])
plt.xlabel('Age')
plt.ylabel('Spending')
plt.title('KMeans Clustering of Age and Spending')
plt.show()