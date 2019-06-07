# Dependencies
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

train_fns = ['AllTaps/240168/240168_session_2.csv', 'AllTaps/240168/240168_session_12.csv', 'AllTaps/240168/240168_session_23.csv', 'AllTaps/240168/240168_session_7.csv', 'AllTaps/240168/240168_session_15.csv', 'AllTaps/240168/240168_session_24.csv', 'AllTaps/240168/240168_session_11.csv', 'AllTaps/240168/240168_session_10.csv', 'AllTaps/240168/240168_session_16.csv', 'AllTaps/240168/240168_session_22.csv', 'AllTaps/240168/240168_session_20.csv', 'AllTaps/240168/240168_session_19.csv']
test_fns = ['AllTaps/240168/240168_session_1.csv', 'AllTaps/240168/240168_session_3.csv', 'AllTaps/240168/240168_session_4.csv', 'AllTaps/240168/240168_session_6.csv', 'AllTaps/240168/240168_session_8.csv', 'AllTaps/240168/240168_session_21.csv', 'AllTaps/240168/240168_session_5.csv', 'AllTaps/240168/240168_session_9.csv', 'AllTaps/240168/240168_session_13.csv', 'AllTaps/240168/240168_session_14.csv', 'AllTaps/240168/240168_session_17.csv', 'AllTaps/240168/240168_session_18.csv']


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

# K-Means Clustering
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train_scaled)

# Make predictions for testing data
predictions = kmeans.predict(X_test_scaled)
kmeans = pd.DataFrame(predictions)

# Visualize results
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(test_data['accelZ_restorationTime'], test_data['gyroZ_restorationTime'], c=kmeans[0])
ax.set_title('K-Means Clustering')
ax.set_xlabel('accelZ_restorationTime')
ax.set_ylabel('gyroZ_restorationTime')
plt.show()