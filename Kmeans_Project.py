# Dependencies
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

train_fns = ['AllTaps/326223/326223_session_1.csv', 'AllTaps/326223/326223_session_2.csv', 'AllTaps/326223/326223_session_3.csv', 'AllTaps/326223/326223_session_6.csv', 'AllTaps/326223/326223_session_7.csv', 'AllTaps/326223/326223_session_8.csv', 'AllTaps/326223/326223_session_12.csv', 'AllTaps/326223/326223_session_10.csv', 'AllTaps/326223/326223_session_13.csv', 'AllTaps/326223/326223_session_15.csv', 'AllTaps/326223/326223_session_19.csv', 'AllTaps/326223/326223_session_22.csv']
test_fns = ['AllTaps/326223/326223_session_5.csv', 'AllTaps/326223/326223_session_9.csv', 'AllTaps/326223/326223_session_4.csv', 'AllTaps/326223/326223_session_11.csv', 'AllTaps/326223/326223_session_17.csv', 'AllTaps/326223/326223_session_24.csv', 'AllTaps/326223/326223_session_18.csv', 'AllTaps/326223/326223_session_16.csv', 'AllTaps/326223/326223_session_14.csv', 'AllTaps/326223/326223_session_21.csv', 'AllTaps/326223/326223_session_20.csv', 'AllTaps/326223/326223_session_23.csv']

train_fns_1 = ['AllTaps/240168/240168_session_2.csv', 'AllTaps/240168/240168_session_12.csv', 'AllTaps/240168/240168_session_23.csv', 'AllTaps/240168/240168_session_7.csv', 'AllTaps/240168/240168_session_15.csv', 'AllTaps/240168/240168_session_24.csv', 'AllTaps/240168/240168_session_11.csv', 'AllTaps/240168/240168_session_10.csv', 'AllTaps/240168/240168_session_16.csv', 'AllTaps/240168/240168_session_22.csv', 'AllTaps/240168/240168_session_20.csv', 'AllTaps/240168/240168_session_19.csv']
test_fns_1 = ['AllTaps/240168/240168_session_1.csv', 'AllTaps/240168/240168_session_3.csv', 'AllTaps/240168/240168_session_4.csv', 'AllTaps/240168/240168_session_6.csv', 'AllTaps/240168/240168_session_8.csv', 'AllTaps/240168/240168_session_21.csv', 'AllTaps/240168/240168_session_5.csv', 'AllTaps/240168/240168_session_9.csv', 'AllTaps/240168/240168_session_13.csv', 'AllTaps/240168/240168_session_14.csv', 'AllTaps/240168/240168_session_17.csv', 'AllTaps/240168/240168_session_18.csv']

train_fns_2 = ['AllTaps/100669/100669_session_1.csv', 'AllTaps/100669/100669_session_2.csv', 'AllTaps/100669/100669_session_3.csv', 'AllTaps/100669/100669_session_4.csv', 'AllTaps/100669/100669_session_5.csv', 'AllTaps/100669/100669_session_7.csv', 'AllTaps/100669/100669_session_9.csv', 'AllTaps/100669/100669_session_10.csv', 'AllTaps/100669/100669_session_14.csv', 'AllTaps/100669/100669_session_16.csv', 'AllTaps/100669/100669_session_17.csv', 'AllTaps/100669/100669_session_22.csv']
test_fns_2 = ['AllTaps/100669/100669_session_11.csv', 'AllTaps/100669/100669_session_8.csv', 'AllTaps/100669/100669_session_18.csv', 'AllTaps/100669/100669_session_6.csv', 'AllTaps/100669/100669_session_12.csv', 'AllTaps/100669/100669_session_15.csv', 'AllTaps/100669/100669_session_19.csv', 'AllTaps/100669/100669_session_13.csv', 'AllTaps/100669/100669_session_23.csv', 'AllTaps/100669/100669_session_20.csv', 'AllTaps/100669/100669_session_21.csv', 'AllTaps/100669/100669_session_24.csv']

train_fns_3 = ['AllTaps/277905/277905_session_1.csv', 'AllTaps/277905/277905_session_2.csv', 'AllTaps/277905/277905_session_5.csv', 'AllTaps/277905/277905_session_6.csv', 'AllTaps/277905/277905_session_8.csv', 'AllTaps/277905/277905_session_9.csv', 'AllTaps/277905/277905_session_12.csv', 'AllTaps/277905/277905_session_13.csv', 'AllTaps/277905/277905_session_14.csv', 'AllTaps/277905/277905_session_15.csv', 'AllTaps/277905/277905_session_17.csv', 'AllTaps/277905/277905_session_22.csv']
test_fns_3 = ['AllTaps/277905/277905_session_4.csv', 'AllTaps/277905/277905_session_3.csv', 'AllTaps/277905/277905_session_10.csv', 'AllTaps/277905/277905_session_7.csv', 'AllTaps/277905/277905_session_11.csv', 'AllTaps/277905/277905_session_16.csv', 'AllTaps/277905/277905_session_20.csv', 'AllTaps/277905/277905_session_21.csv', 'AllTaps/277905/277905_session_19.csv', 'AllTaps/277905/277905_session_23.csv', 'AllTaps/277905/277905_session_18.csv', 'AllTaps/277905/277905_session_24.csv']

train_fns_4 = ['AllTaps/201848/201848_session_1.csv', 'AllTaps/201848/201848_session_2.csv', 'AllTaps/201848/201848_session_3.csv', 'AllTaps/201848/201848_session_4.csv', 'AllTaps/201848/201848_session_6.csv', 'AllTaps/201848/201848_session_7.csv', 'AllTaps/201848/201848_session_11.csv', 'AllTaps/201848/201848_session_15.csv', 'AllTaps/201848/201848_session_16.csv', 'AllTaps/201848/201848_session_18.csv', 'AllTaps/201848/201848_session_19.csv', 'AllTaps/201848/201848_session_21.csv']
test_fns_4 = ['AllTaps/201848/201848_session_13.csv', 'AllTaps/201848/201848_session_10.csv', 'AllTaps/201848/201848_session_5.csv', 'AllTaps/201848/201848_session_8.csv', 'AllTaps/201848/201848_session_9.csv', 'AllTaps/201848/201848_session_14.csv', 'AllTaps/201848/201848_session_12.csv', 'AllTaps/201848/201848_session_24.csv', 'AllTaps/201848/201848_session_17.csv', 'AllTaps/201848/201848_session_22.csv', 'AllTaps/201848/201848_session_20.csv', 'AllTaps/201848/201848_session_23.csv']

train_fns_5 = []
test_fns_5 = []

def getDataFrame(filenames):
    length = len(filenames)
    data = pd.read_csv(filenames[0])

    for i in range(1, length):
        new_data = pd.read_csv(filenames[i])
        data = pd.concat([data, new_data], ignore_index=True) 
    return data       

train_data = getDataFrame(train_fns_5)
test_data = getDataFrame(test_fns_5)

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

# PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.fit_transform(X_test_scaled)

# Isomap
embedding = Isomap(n_components=2)
#X_train_iso = embedding.fit_transform(X_train_scaled)
#X_test_iso = embedding.fit_transform(X_test_scaled)

# K-Means Clustering
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train_pca)

# Make predictions for testing data
predictions = kmeans.predict(X_test_pca)
kmeans = pd.DataFrame(predictions)

# Plot 2-Dimensional Data acquired from PCA/Isomap
test_df = pd.DataFrame(X_test_pca)
plt.rcParams.update({'font.size': 30})
fig_1 = plt.figure(1)
ax = fig_1.add_subplot(111)
scatter = ax.scatter(test_df[0], test_df[1], c=kmeans[0])
ax.set_title('User 151985 PCA Results')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')

# Visualize results
plt.rcParams.update({'font.size': 30})
fig_2 = plt.figure(2)
ax = fig_2.add_subplot(111)
scatter = ax.scatter(test_data['accelZ_restorationTime'], test_data['gyroZ_restorationTime'], c=kmeans[0])
ax.set_title('User 151985 PCA K-Means Clustering')
ax.set_xlabel('accelZ_restorationTime')
ax.set_ylabel('gyroZ_restorationTime')
plt.show()