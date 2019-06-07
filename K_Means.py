# Dependencies
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Practice using K-Means
# Tutorial from: https://www.datacamp.com/community/tutorials/k-means-clustering-python#case

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Fill missing values with mean column values in the train set
train_data.fillna(train_data.mean(), inplace=True)

# Fill missing values with mean column values in the test set
test_data.fillna(test_data.mean(), inplace=True)

# Drop unecessary columns
train_data = train_data.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)
test_data = test_data.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)

# encode Sex column with numerical values
labelEncoder = LabelEncoder()
labelEncoder.fit(train_data['Sex'])
labelEncoder.fit(test_data['Sex'])
train_data['Sex'] = labelEncoder.transform(train_data['Sex'])
test_data['Sex'] = labelEncoder.transform(test_data['Sex'])

print(train_data.head())
# Drop Survived column(unsupervised)
X = np.array(train_data.drop(['Survived'], 1).astype(float))

# Save Survived column values for checking
y = np.array(train_data['Survived'])

# kmeans = KMeans(n_clusters=2) # You want cluster the passenger records into 2: Survived or Not survived
# kmeans.fit(X)

kmeans = kmeans = KMeans(n_clusters=2, max_iter=600, algorithm = 'auto')
kmeans.fit(X)

# Scaling the values of the features to the same range
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
kmeans.fit(X_scaled)

def predictions(features, k_means):
    length = len(features)
    output = []
    
    for i in range(length):
        predict_me = np.array(features[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))
        prediction = k_means.predict(predict_me)

        output.append(prediction[0])

    return output

def test_accuracy(features, clusters, k_means):
    correct = 0
    length = len(features)

    for i in range(length):
        predict_me = np.array(features[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))
        prediction = kmeans.predict(predict_me)
        
        if prediction[0] == clusters[i]:
            correct += 1
    
    return correct/length

######################################################################
# Practice visualizing data
# Tutorial from: https://www.kaggle.com/dhanyajothimani/basic-visualization-and-clustering-in-python/notebook

wh = pd.read_csv("2017.csv")
wh1 = wh[['Happiness.Score','Economy..GDP.per.Capita.','Family','Health..Life.Expectancy.', 'Freedom', 
          'Generosity','Trust..Government.Corruption.','Dystopia.Residual']]

#K means Clustering 
def do_kmeans(X, nclust=2):
    model = KMeans(nclust)
    model.fit(X)
    clust_labels = model.predict(X)
    cent = model.cluster_centers_
    return (clust_labels, cent)

clust_labels, cent = do_kmeans(wh1, 2)
kmeans = pd.DataFrame(clust_labels)
wh1.insert((wh1.shape[1]),'kmeans',kmeans)

#Plot the clusters obtained using k means
"""fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(wh1['Economy..GDP.per.Capita.'],wh1['Trust..Government.Corruption.'],
                     c=kmeans[0],s=50)
ax.set_title('K-Means Clustering')
ax.set_xlabel('GDP per Capita')
ax.set_ylabel('Corruption')
plt.colorbar(scatter)
plt.show()"""