import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])


colors = ['r','g','b','c','k','o','y']

train_fns = ['AllTaps/326223/326223_session_1.csv', 'AllTaps/326223/326223_session_2.csv', 'AllTaps/326223/326223_session_3.csv', 'AllTaps/326223/326223_session_6.csv', 'AllTaps/326223/326223_session_7.csv', 'AllTaps/326223/326223_session_8.csv', 'AllTaps/326223/326223_session_12.csv', 'AllTaps/326223/326223_session_10.csv', 'AllTaps/326223/326223_session_13.csv', 'AllTaps/326223/326223_session_15.csv', 'AllTaps/326223/326223_session_19.csv', 'AllTaps/326223/326223_session_22.csv']
test_fns = ['AllTaps/326223/326223_session_5.csv', 'AllTaps/326223/326223_session_9.csv', 'AllTaps/326223/326223_session_4.csv', 'AllTaps/326223/326223_session_11.csv', 'AllTaps/326223/326223_session_17.csv', 'AllTaps/326223/326223_session_24.csv', 'AllTaps/326223/326223_session_18.csv', 'AllTaps/326223/326223_session_16.csv', 'AllTaps/326223/326223_session_14.csv', 'AllTaps/326223/326223_session_21.csv', 'AllTaps/326223/326223_session_20.csv', 'AllTaps/326223/326223_session_23.csv']

def getDataFrame(filenames):
    length = len(filenames)
    data = pd.read_csv(filenames[0])

    for i in range(1, length):
        new_data = pd.read_csv(filenames[i])
        data = pd.concat([data, new_data], ignore_index=True) 
    return data       

train_data = getDataFrame(train_fns)
test_data = getDataFrame(test_fns)

# Fill missing values with mean column values in the train set and the test set
train_data.fillna(train_data.mean(), inplace=True)
test_data.fillna(test_data.mean(), inplace=True)

# Drop unecessary columns
train_data = train_data.drop(['UserID', 'Session', 'TaskName'], axis=1)
test_data = test_data.drop(['UserID', 'Session', 'TaskName'], axis=1)

# Replace inf values with mean column values in the train set and test set
train_data[train_data==np.inf]=np.nan
train_data.fillna(train_data.mean(), inplace=True)
test_data[test_data==np.inf]=np.nan
test_data.fillna(test_data.mean(), inplace=True)

# Convert DataFrame to numpy array
X_train = np.array(train_data.drop(['Orientation'], 1).astype(float))
X_test = np.array(test_data.drop(['Orientation'], 1).astype(float))

# Scale to values between 0 and 1 for accuracy
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


# Custom K-Means tutorial found on https://pythonprogramming.net/k-means-from-scratch-machine-learning-tutorial/
class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    #print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break

            return self.centroids, self.classifications

    def predict(self,data):
        predictions = []
        for featureset in data:
            distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            prediction = distances.index(min(distances))
            predictions.append(prediction)
        return predictions

k_means = K_Means()
centroids, classifications = k_means.fit(X_train_scaled)
predictions = k_means.predict(X_test_scaled)
kmeans = pd.DataFrame(predictions)

# Visualize results
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(test_data['accelZ_restorationTime'], test_data['gyroZ_restorationTime'], c=kmeans[0])
ax.set_title('User 326223 K-Means Clustering')
ax.set_xlabel('accelZ_restorationTime')
ax.set_ylabel('gyroZ_restorationTime')
plt.show()