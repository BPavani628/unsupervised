import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#step 1: Load the data
df= pd.read_csv('growth_data_2.csv',na_values=['null'])
print(df)
df = df.fillna(0)
print(df)
#step 2: Selecting the column data
X = df[['Year','Growth']]
print(X)
train = KMeans(df)
#step 3: Using kmeans to cluster the data
model = KMeans(n_clusters=7, random_state=0)
model.fit(X)
df['cluster'] = model.labels_
#step 4: Visualizing the clusters
plt.scatter(df['Year'], df['Growth'], c=df['cluster'])
plt.xlabel('Year')
plt.ylabel('Growth')
plt.title('KMeans Clustering of Year and Growth')
plt.show()


