# Dependencies
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
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

train_fns_5 = ['AllTaps/261313/261313_session_2.csv', 'AllTaps/261313/261313_session_3.csv', 'AllTaps/261313/261313_session_4.csv', 'AllTaps/261313/261313_session_5.csv', 'AllTaps/261313/261313_session_6.csv', 'AllTaps/261313/261313_session_7.csv', 'AllTaps/261313/261313_session_11.csv','AllTaps/261313/261313_session_13.csv', 'AllTaps/261313/261313_session_15.csv','AllTaps/261313/261313_session_18.csv', 'AllTaps/261313/261313_session_23.csv']
test_fns_5 = ['AllTaps/261313/261313_session_8.csv', 'AllTaps/261313/261313_session_10.csv', 'AllTaps/261313/261313_session_19.csv', 'AllTaps/261313/261313_session_9.csv', 'AllTaps/261313/261313_session_12.csv', 'AllTaps/261313/261313_session_17.csv', 'AllTaps/261313/261313_session_20.csv', 'AllTaps/261313/261313_session_16.csv', 'AllTaps/261313/261313_session_21.csv', 'AllTaps/261313/261313_session_22.csv', 'AllTaps/261313/261313_session_24.csv']


def getDataFrame(filenames):
    length = len(filenames)
    data = pd.read_csv(filenames[0])

    for i in range(1, length):
        new_data = pd.read_csv(filenames[i])
        data = pd.concat([data, new_data], ignore_index=True) 
    return data       

def preprocessData(train_fn, test_fn):
    train_data = getDataFrame(train_fn)
    test_data = getDataFrame(test_fn)

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
    X_data = np.concatenate((X_train, X_test))

    # Scale values
    #scaler = MinMaxScaler()
    scaler = StandardScaler()
    scaler.fit(X_data)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return train_data, test_data, X_train_scaled, X_test_scaled

def pca(X_train_scaled, X_test_scaled):
    # PCA
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.fit_transform(X_test_scaled)

    return X_train_pca, X_test_pca

def isomap(X_train_scaled, X_test_scaled):
    # Isomap
    embedding = Isomap(n_components=2)
    X_train_iso = embedding.fit_transform(X_train_scaled)
    X_test_iso = embedding.fit_transform(X_test_scaled)

    return X_train_iso, X_test_iso

def predictions(training, testing):
    # K-Means Clustering
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(training)

    # Make predictions for testing data
    predictions = kmeans.predict(testing)
    kmeans_predictions = pd.DataFrame(predictions)

    return kmeans_predictions

def visualizePCAResults(user, kmeans_predictions, test_data):
    # Plot 2-Dimensional Data acquired from PCA
    test_df = pd.DataFrame(test_data)
    plt.rcParams.update({'font.size': 30})
    fig_1 = plt.figure()
    ax = fig_1.add_subplot(111)
    scatter = ax.scatter(test_df[0], test_df[1], c=kmeans_predictions[0])
    title = 'User ' + user + ' PCA Results'
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    plt.show()

def visualizeIsomapResults(user, kmeans_predictions, test_data):
    # Plot 2-Dimensional Data acquired from Isomap
    test_df = pd.DataFrame(test_data)
    plt.rcParams.update({'font.size': 30})
    fig_1 = plt.figure()
    ax = fig_1.add_subplot(111)
    scatter = ax.scatter(test_df[0], test_df[1], c=kmeans_predictions[0])
    title = 'User ' + user + ' Isomap Results'
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    plt.show()

def visualizeKmeans(user, kmeans_predictions, test_data):
    # Visualize results
    plt.rcParams.update({'font.size': 30})
    fig_2 = plt.figure()
    ax = fig_2.add_subplot(111)
    scatter = ax.scatter(test_data['accelZ_restorationTime'], test_data['gyroZ_restorationTime'], c=kmeans_predictions[0])
    title = 'User ' + user + ' K-Means Clustering'
    ax.set_title(title)
    ax.set_xlabel('accelZ_restorationTime')
    ax.set_ylabel('gyroZ_restorationTime')
    plt.show()

'''
def main(user, train_fns, test_fns):
    train_data, test_data, X_train_scaled, X_test_scaled = preprocessData(train_fns, test_fns)
    X_train_pca, X_test_pca = pca(X_train_scaled, X_test_scaled)
    X_train_iso, X_test_iso = isomap(X_train_scaled, X_test_scaled)

    # Predictions
    kmeans_predictions = predictions(X_train_scaled, X_test_scaled)
    kmeans_predictions_pca = predictions(X_train_pca, X_test_pca)
    kmeans_predictions_iso = predictions(X_train_iso, X_test_iso)

    # Visualize results
    visualizeKmeans(user, kmeans_predictions, test_data)
    visualizePCAResults(user , kmeans_predictions_pca, X_test_pca)
    visualizeIsomapResults(user , kmeans_predictions_iso, X_test_iso)
    '''