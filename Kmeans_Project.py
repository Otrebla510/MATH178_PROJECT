# Dependencies
import glob
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


users = ['100669', '201848', '240168', '277905', '326223', '368258', '389015', '395129', '396697', '405035', '431312', '527796', '538363', '579284', '594887', '621276', '622852', '710707', '720193', '745224', '751131', '763813', '776328', '796581', '803262', '808022', '872895', '879155', '893255', '897652', '962159', '973891']


def getFilenames(user):
    train_ms = 0
    train_rs = 0
    train_ws = 0
    train_mw = 0
    train_rw = 0
    train_ww = 0

    test_ms = 0
    test_rs = 0
    test_ws = 0
    test_mw = 0
    test_rw = 0
    test_ww = 0

    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    path = 'AllTaps/' + user + '/*.csv' 
    for fname in glob.glob(path):
        data = pd.read_csv(fname)
        if data['TaskName'][0] == 'Map+Sitting':
            if test_ms < train_ms:
                test_data = test_data.append(data, ignore_index=True)
                test_ms += 1
            else:
                train_data = train_data.append(data, ignore_index=True)
                train_ms += 1
        if data['TaskName'][0] == 'Reading+Sitting':
            if test_rs < train_rs:
                test_data = test_data.append(data, ignore_index=True)
                test_rs += 1
            else:
                train_data = train_data.append(data, ignore_index=True)
                train_rs += 1
        if data['TaskName'][0] == 'Writing+Sitting':
            if test_ws < train_ws:
                test_data = test_data.append(data, ignore_index=True)
                test_ws += 1
            else:
                train_data = train_data.append(data, ignore_index=True)
                train_ws += 1
        if data['TaskName'][0] == 'Map+Walking':
            if test_mw < train_mw:
                test_data = test_data.append(data, ignore_index=True)
                test_mw += 1
            else:
                train_data = train_data.append(data, ignore_index=True)
                train_mw += 1
        if data['TaskName'][0] == 'Reading+Walking':
            if test_rw < train_rw:
                test_data = test_data.append(data, ignore_index=True)
                test_rw += 1
            else:
                train_data = train_data.append(data, ignore_index=True)
                train_rw += 1
        if data['TaskName'][0] == 'Writing+Walking':
            if test_ww < train_ww:
                test_data = test_data.append(data, ignore_index=True)
                test_ww += 1
            else:
                train_data = train_data.append(data, ignore_index=True)
                train_ww += 1
    '''
    print('Training------')
    print('MS: ', train_ms)
    print('RS: ', train_rs)
    print('WS: ', train_ws)
    print('MW: ', train_mw)
    print('RW: ', train_rw)
    print('WW: ', train_ww)

    print('Testing------')
    print('MS: ', test_ms)
    print('RS: ', test_rs)
    print('WS: ', test_ws)
    print('MW: ', test_mw)
    print('RW: ', test_rw)
    print('WW: ', test_ww)
    '''

    return train_data, test_data

def preprocessData(user):
    train_data, test_data = getFilenames(user)

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
    scaler = StandardScaler()
    scaler.fit(X_data)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return train_data, test_data, X_train_scaled, X_test_scaled

def pca(X_train_scaled, X_test_scaled, num_components):
    # PCA
    pca = PCA(n_components=num_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.fit_transform(X_test_scaled)

    return X_train_pca, X_test_pca

def isomap(X_train_scaled, X_test_scaled, num_components):
    # Isomap
    embedding = Isomap(n_components=num_components)
    X_train_iso = embedding.fit_transform(X_train_scaled)
    X_test_iso = embedding.fit_transform(X_test_scaled)

    return X_train_iso, X_test_iso

def se(X_train_scaled, X_test_scaled, num_components):
    # Locally Linear Embedding
    embedding = SpectralEmbedding(n_components=num_components)
    X_train_se = embedding.fit_transform(X_train_scaled)
    X_test_se = embedding.fit_transform(X_test_scaled)

    return X_train_se, X_test_se

def predictions(training, testing):
    # K-Means Clustering
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(training)

    # Make predictions for testing data
    predictions = kmeans.predict(testing)
    kmeans_predictions = pd.DataFrame(predictions)

    return kmeans_predictions

def visualize2DPCAResults(user, kmeans_predictions, test_data):
    # Plot 2-Dimensional Data acquired from PCA
    test_df = pd.DataFrame(test_data)
    #plt.rcParams.update({'font.size': 30})
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(test_df[0], test_df[1], c=kmeans_predictions[0])
    title = 'User ' + user + ' 2D PCA Results'
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')

    plot_name = user + '_pca2D_results.png'
    save_plot = 'Visualizations/User ' + user + '/' + plot_name
    fig.savefig(save_plot, bbox_inches='tight')
    #plt.show()

def visualize3DPCAResults(user, kmeans_predictions, test_data):
    # Plot 3-Dimensional Data acquired from PCA
    test_df = pd.DataFrame(test_data)
    fig = plt.figure()
    #plt.rcParams.update({'font.size': 30})
    ax = plt.axes(projection='3d')
    ax.scatter3D(test_df[0], test_df[1], test_df[2], c=kmeans_predictions[0])
    title = 'User ' + user + ' 3D PCA Results'
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')

    plot_name = user + '_pca3D_results.png'
    save_plot = 'Visualizations/User ' + user + '/' + plot_name
    fig.savefig(save_plot, bbox_inches='tight')
    #plt.show()

def visualize2DIsomapResults(user, kmeans_predictions, test_data):
    # Plot 2-Dimensional Data acquired from Isomap
    test_df = pd.DataFrame(test_data)
    #plt.rcParams.update({'font.size': 30})
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(test_df[0], test_df[1], c=kmeans_predictions[0])
    title = 'User ' + user + ' 2D Isomap Results'
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')

    plot_name = user + '_isomap2D_results.png'
    save_plot = 'Visualizations/User ' + user + '/' + plot_name
    fig.savefig(save_plot, bbox_inches='tight')
    #plt.show()

def visualize3DIsomapResults(user, kmeans_predictions, test_data):
    # Plot 3-Dimensional Data acquired from PCA
    test_df = pd.DataFrame(test_data)
    fig = plt.figure()
    #plt.rcParams.update({'font.size': 30})
    ax = plt.axes(projection='3d')
    ax.scatter3D(test_df[0], test_df[1], test_df[2], c=kmeans_predictions[0])
    title = 'User ' + user + ' 3D Isomap Results'
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')

    plot_name = user + '_isomap3D_results.png'
    save_plot = 'Visualizations/User ' + user + '/' + plot_name
    fig.savefig(save_plot, bbox_inches='tight')
    #plt.show()

def visualize2DSEResults(user, kmeans_predictions, test_data):
    # Plot 2-Dimensional Data acquired from Isomap
    test_df = pd.DataFrame(test_data)
    #plt.rcParams.update({'font.size': 30})
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(test_df[0], test_df[1], c=kmeans_predictions[0])
    title = 'User ' + user + ' 2D Spectral Embedding Results'
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')

    plot_name = user + '_spectral2D_results.png'
    save_plot = 'Visualizations/User ' + user + '/' + plot_name
    fig.savefig(save_plot, bbox_inches='tight')
    #plt.show()

def visualize3DSEResults(user, kmeans_predictions, test_data):
    # Plot 3-Dimensional Data acquired from PCA
    test_df = pd.DataFrame(test_data)
    fig = plt.figure()
    #plt.rcParams.update({'font.size': 30})
    ax = plt.axes(projection='3d')
    ax.scatter3D(test_df[0], test_df[1], test_df[2], c=kmeans_predictions[0])
    title = 'User ' + user + ' 3D Spectral Embedding Results'
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')

    plot_name = user + '_spectral3D_results.png'
    save_plot = 'Visualizations/User ' + user + '/' + plot_name
    fig.savefig(save_plot, bbox_inches='tight')
    #plt.show()

def visualizeKmeans(user, kmeans_predictions, test_data, method, method_name, dimensions):
    # Visualize results
    #plt.rcParams.update({'font.size': 30})
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(test_data['accelZ_restorationTime'], test_data['gyroZ_restorationTime'], c=kmeans_predictions[0])
    title = 'User ' + user + ' ' + method + 'K-Means Clustering'
    ax.set_title(title)
    ax.set_xlabel('accelZ_restorationTime')
    ax.set_ylabel('gyroZ_restorationTime')

    if method_name == '':
        plot_name = user + 'z_kmeans.png'
    else:
        if dimensions == 2:
            plot_name = user + 'z_' + method_name + '2D_kmeans.png'
        if dimensions == 3:
            plot_name = user + 'z_' + method_name + '3D_kmeans.png'
    save_plot = 'Visualizations/User ' + user + '/' + plot_name
    fig.savefig(save_plot, bbox_inches='tight')
    #plt.show()

def visualize(user):
    # Preprocess Data
    train_data, test_data, X_train_scaled, X_test_scaled = preprocessData(user)
    X_train_pca_2, X_test_pca_2 = pca(X_train_scaled, X_test_scaled, 2)
    X_train_iso_2, X_test_iso_2 = isomap(X_train_scaled, X_test_scaled, 2)
    X_train_se_2, X_test_se_2 = se(X_train_scaled, X_test_scaled, 2)
    X_train_pca_3, X_test_pca_3 = pca(X_train_scaled, X_test_scaled, 3)
    X_train_iso_3, X_test_iso_3 = isomap(X_train_scaled, X_test_scaled, 3)
    X_train_se_3, X_test_se_3 = se(X_train_scaled, X_test_scaled, 3)

    # Predictions
    kmeans_predictions = predictions(X_train_scaled, X_test_scaled)
    kmeans_predictions_pca_2 = predictions(X_train_pca_2, X_test_pca_2)
    kmeans_predictions_iso_2 = predictions(X_train_iso_2, X_test_iso_2)
    kmeans_predictions_se_2 = predictions(X_train_se_2, X_test_se_2)
    kmeans_predictions_pca_3 = predictions(X_train_pca_3, X_test_pca_3)
    kmeans_predictions_iso_3 = predictions(X_train_iso_3, X_test_iso_3)
    kmeans_predictions_se_3 = predictions(X_train_se_3, X_test_se_3)

    # Visualize Kmeans results
    visualizeKmeans(user, kmeans_predictions, test_data, '', '', 0)
    visualizeKmeans(user, kmeans_predictions_pca_2, test_data, '2D PCA ', 'pca', 2)
    visualizeKmeans(user, kmeans_predictions_iso_2, test_data, '2D Isomap ', 'isomap', 2)
    visualizeKmeans(user, kmeans_predictions_se_2, test_data, '2D Spectral Embedding ', 'spectral', 2)
    visualizeKmeans(user, kmeans_predictions_pca_3, test_data, '3D PCA ', 'pca', 3)
    visualizeKmeans(user, kmeans_predictions_iso_3, test_data, '3D Isomap ', 'isomap', 3)
    visualizeKmeans(user, kmeans_predictions_se_3, test_data, '3D Spectral Embedding ', 'spectral', 3)

    # 2D Visualization
    visualize2DPCAResults(user, kmeans_predictions_pca_2, X_test_pca_2)
    visualize2DIsomapResults(user, kmeans_predictions_iso_2, X_test_iso_2)
    visualize2DSEResults(user, kmeans_predictions_se_2, X_test_se_2)
    
    # 3D Visualization
    visualize3DPCAResults(user, kmeans_predictions_pca_3, X_test_pca_3)
    visualize3DIsomapResults(user, kmeans_predictions_iso_3, X_test_iso_3)
    visualize3DSEResults(user, kmeans_predictions_se_3, X_test_se_3)

def main(users):
    for i in range(18, 33):
        visualize(users[i])
        string = 'Done with ' + users[i]
        print(string)

main(users)