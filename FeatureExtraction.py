
from pathlib import Path
from typing import List
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statistics import mode
from statistics import mean
from statistics import stdev as sd
import scipy.stats as stat

import os
import csv
import random
import numpy as np

# Data pre processing
base_data_folder_path = Path('public_dataset')
file_name_to_column_names = {
    'Accelerometer.csv': ['Systime', 'EventTime', 'ActivityID', 'X', 'Y', 'Z', 'Phone_orientation'],
    'Activity.csv': ['ID', 'SubjectID', 'Start_time', 'End_time', 'Relative_Start_time', 'Relative_End_time',
                     'Gesture_scenario', 'TaskID', 'ContentID'],
    'Gyroscope.csv': ['Systime_gyro', 'EventTime', 'ActivityID', 'X', 'Y', 'Z', 'Phone_orientation'],
    'TouchEvent.csv': ['Systime', 'EventTime', 'ActivityID', 'PointerCount', 'PointerID', \
                       'ActionID', 'X', 'Y', 'Pressure', 'Contact_size', 'Phone_orientation'],
    'KeyPressEvent.csv': ['Systime', 'PressTime', 'PressType', 'ActivityID', 'KeyID', 'Phone_orientation'],
    'OneFingerTouchEvent.csv': ['Systime', 'PressTime', 'ActivityID', 'TapID', 'Tap_type', \
                                'Action_type', 'X', 'Y', 'Pressure', 'Contact_size', 'Phone_orientation'],
    'PinchEvent.csv': ['Systime', 'PressTime', 'ActivityID', 'EventType', 'PinchID', 'Time_delta', 'Focus_X', \
                       'Focus_Y', 'Span', 'Span_X', 'Span_Y', 'ScaleFactor', 'Phone_orientation'],
    'ScrollEvent.csv': ['Systime', 'BeginTime', 'CurrentTime', 'ActivityID', 'ScrollID', \
                        'Start_action_type', 'Start_X', 'Start_Y', 'Start_pressure', \
                        'Start_size', 'Current_action_type', 'Current_X', 'Current_Y', \
                        'Current_pressure', 'Current_size', 'Distance_X', 'Distance_Y', \
                        'Phone_orientation'],
    'StrokeEvent.csv': ['Systime', 'Begin_time', 'End_time', 'ActivityID', 'Start_action_type', \
                        'Start_X', 'Start_Y', 'Start_pressure', 'Start_size', 'End_action_type', \
                        'End_X', 'End_Y', 'End_pressure', 'End_size', 'Speed_X', 'Speed_Y', \
                        'Phone_orientation']
}
x_columns = {
    'Accelerometer.csv': 'EventTime_accel',
    'Gyroscope.csv': 'EventTime_gyro'
}
y_columns = {
    'Accelerometer.csv': ['X_accel', 'Y_accel', 'Z_accel', 'M_accel'],
    'Gyroscope.csv': ['X_gyro', 'Y_gyro', 'Z_gyro', 'M_gyro']
}
other_columns = {
    'Accelerometer.csv': ['Phone_orientation_accel'],
    'Gyroscope.csv': ['Phone_orientation_gyro']
}
tap_file_names = [
    'TouchEvent.csv',
    'KeyPressEvent.csv',
    'OneFingerTouchEvent.csv',
    'PinchEvent.csv',
    'ScrollEvent.csv',
    'StrokeEvent.csv']
tap_file_to_feature_name = {
    'TouchEvent.csv': 'Touch',
    'KeyPressEvent.csv': 'KeyPress',
    'OneFingerTouchEvent.csv': 'OneFingerTouch',
    'PinchEvent.csv': 'Pinch',
    'ScrollEvent.csv': 'Scroll',
    'StrokeEvent.csv': 'Stroke'
}
tap_feature_to_file_name = {
    'Touch': 'TouchEvent.csv',
    'KeyPress': 'KeyPressEvent.csv',
    'OneFingerTouch': 'OneFingerTouchEvent.csv',
    'Pinch': 'PinchEvent.csv',
    'Scroll': 'ScrollEvent.csv',
    'Stroke': 'StrokeEvent.csv' ,
}
tap_file_important_columns = {
    'TouchEvent.csv': ['EventTime', 'Phone_orientation'],
    'KeyPressEvent.csv': ['PressTime', 'Phone_orientation'],
    'OneFingerTouchEvent.csv': ['PressTime', 'Phone_orientation'],
    'PinchEvent.csv': ['PressTime', 'Phone_orientation'],
    'ScrollEvent.csv': ['BeginTime', 'CurrentTime', 'Phone_orientation'],
    'StrokeEvent.csv': ['Begin_time', 'End_time', 'Phone_orientation']
}
file_names = ['Accelerometer.csv', 'Gyroscope.csv']

task_names = ["Reading+Sitting", "Reading+Walking", "Writing+Sitting", \
    "Writing+Walking", "Map+Sitting", "Map+Walking"]

labels = ['UserID', 'Session', 'TaskName', 'Orientation']
for file_name in file_names:
    for y in y_columns[file_name]:
        labels += [y+'_mean', y+'_stdev', y+'_differenceBeforeAfter', \
            y+'_netChange', y+'_maxChange', y+'_restorationTime', \
            y+'_normedDuration', y+'_normedDurationMax']


# Here are the features for shape vectors, which are used to measure the
# the relatedness of different histograms
MIN_X = -1e8
MAX_X = 1e8  # These define the region we consider; based on observations these should be good,
MIN_Y = -0.8   # in the sense that they enclose most of the distributions pretty tightly
MAX_Y = 0.8    # Change of plans - next time you run it, change the y bounds to 0.8 and -0.8
BIN_NUMBER = 40 # The region of the plane is divided into BIN_NUMBER x BIN_NUMBER rectangles

THRESHOLD = 50

def get_user_ids() -> List[str]:
    """
    Get all user ids based on name of folders under "public_dataset/"
    :return: a list of user ids
    """
    listOfFiles = os.listdir('public_dataset')
    listOfFiles.remove('data_description.pdf')
    try:
        listOfFiles.remove('.DS_Store')
    except:
        pass
    return listOfFiles
    
def get_user_session_ids(user_id: str) -> List[str]:
    """
    Get all session ids for a specific user based on folder structure
    e.g. "public_dataset/100669/100669_session_13" has user_id=100669, session_id=13
    :param user_id: user id
    :return: list of user session ids
    """
    listOfSessions = os.listdir('public_dataset/'+user_id)
    try:
        listOfSessions.remove('.DS_Store')
    except:
        pass
    return listOfSessions

def get_user_session_ids_for_task(user_id: str, task_name: str) -> List[str]:
    """
    Get all session ids for a specific user and task based on folder structure
    e.g. "public_dataset/100669/100669_session_13" has user_id=100669, session_id=13
    :param user_id: user id
    :return: list of user session ids
    """
    listOfSessions = os.listdir('Plots/Research/'+user_id+'/'+task_name)
    try:
        listOfSessions.remove('.DS_Store')
    except:
        pass
    return listOfSessions

def read_file(user_id: str, user_session_id: str, file_name: str) -> DataFrame:
    """
    Read one of the csv files for a user
    :param user_id: user id
    :param user_session_id: user session id
    :param file_name: csv file name (key of file_name_to_column_names)
    :param column_names: a list of column names of the csv file (subset of what's in file_name_to_column_names)
    :return: content of the csv file as pandas DataFrame
    """
    path = 'public_dataset/'+user_id+'/'+user_session_id+'/'+file_name
    full_columns = file_name_to_column_names[file_name]
    data_frame = pd.read_csv(path, header=None, names = file_name_to_column_names[file_name], index_col=False)
    return data_frame

def get_user_session_data(user_id: str, user_session_id: str) -> DataFrame:
    """
    Combine accelerometer, gyroscope, and activity labels for a specific session of a user
    Note: Timestamps are ignored when joining accelerometer and gyroscope data.
    :param user_id: user id
    :param user_session_id: user session id
    :return: combined DataFrame for a session
    """
    activity_df = read_file(user_id, user_session_id, 'Activity.csv')
    accel_df = read_file(user_id, user_session_id, 'Accelerometer.csv')
    gyro_df = read_file(user_id, user_session_id, 'Gyroscope.csv')

    measurements_df = accel_df.join(gyro_df, lsuffix = '_accel', rsuffix = '_gyro')
    full_df = measurements_df.join(activity_df.set_index('ID'), on='ActivityID' + '_accel')
    full_df = full_df.dropna().reset_index(drop = True)

    return full_df

def add_magnitude_columns(data: DataFrame):
    """
    Adds the Euclidean norm of the accelerometer and gyroscope data
    """
    data['M_accel'] = data[['X_accel','Y_accel','Z_accel']].apply(np.linalg.norm, axis = 1)
    data['M_gyro'] = data[['X_gyro','Y_gyro','Z_gyro']].apply(np.linalg.norm, axis = 1)


def get_tap_events(user_id: str, user_session_id: str) -> DataFrame:
    """
    Return a dataframe of the relative times at which tap events occur.
    :param user_id: user id
    :param user_session_id: user session id
    :param windows: the time frames during which to look for tap events
    :return: A list of the tap events, in the format [beginTime, endTime]
    """
    full_df = pd.DataFrame()
    for tap_file in tap_file_names:
        columns = tap_file_important_columns[tap_file]
        data = read_file(user_id, user_session_id, tap_file)
        time_data = pd.DataFrame()
        time_data['Start'] = data[columns[0]]
        time_data['End'] = data[columns[-2]]
        time_data['Type'] = tap_file_to_feature_name[tap_file]
        full_df = pd.concat([full_df, time_data], ignore_index = True)
    return full_df.dropna().sort_values(by = 'Start').reset_index(drop = True)

def add_columns_for_taps(full_data, tap_data):
    for tap_file in tap_file_names:
        tap_type = tap_file_to_feature_name[tap_file]
        data = tap_data[tap_data['Type'] == tap_type].reset_index(drop = True)

        lead_file = 'Accelerometer.csv'
        time_column_name = x_columns[lead_file]
        data_times = full_data[time_column_name]
        data_index = 0

        new_column = []

        for tap_index in range(data.shape[0]):
            try:
                while data_times[data_index] < (data['Start'][tap_index] * 1000000):
                    new_column.append(0) # Not in the midst of a tap
                    data_index += 1
                    if data_index >= full_data.shape[0]: break
                if data_index >= full_data.shape[0]: break
                new_column.append(1) # At least one value in the midst of the tap
                data_index += 1
                if data_index >= full_data.shape[0]: break
                while data_times[data_index] < (data['End'][tap_index] * 1000000):
                    new_column.append(1)
                    data_index += 1
                    if data_index >= full_data.shape[0]: break
                if data_index >= full_data.shape[0]: break
            except KeyError:
                print("Okay, here's that thing again")
                return

            
        while data_index < full_data.shape[0]:
            new_column.append(0)
            data_index += 1

        full_data[tap_type] = new_column
        
def mark_tap_start_and_end(data: DataFrame, delta_in_ms: int):
    """
    Locates each individual tap event and puts special values in
    the column to indicate the start and end
    :param data: The data, preprocessed by add_columns_for_taps
    :param delta: The distance between taps necessary for taps to be distinct
    """

    lead_file = 'Accelerometer.csv'
    time_col = x_columns[lead_file]

    delta = delta_in_ms * 1000000

    for tap_file in tap_file_names:
        tap_feature = tap_file_to_feature_name[tap_file]
        # Step 1: Put a 2 at the start and a 3 at the end of each event

        indices = data[data[tap_feature] == 1].index
        if len(indices) == 0:
            continue
        for i in range(len(indices)):
            if i == 0 or data[time_col][ indices[i] ] - data[time_col][ indices[i - 1] ] > delta:
                data[tap_feature].loc[ indices[i] ] = 2
                if i > 0:
                    if data[tap_feature][ indices[i - 1] ] == 1:
                        data[tap_feature].loc[ indices[i - 1] ] = 3
                    elif indices[i - 1] + 1 < data.shape[0] and data[tap_feature][ indices[i - 1] + 1 ] == 0:
                        # In this case, the tap lasted only one time step,
                        # so we call the end of the last tap the reading after
                        data[tap_feature].loc[ indices[i - 1] + 1 ] = 3
                    else:
                        #Hopefully this case will never occur, where two consecutive taps
                        #are more than delta apart but with no readings in between
                        print("Something seems off about this data...")
                        print(data[ indices[i] - 5 : indices[i] + 5][[time_col, tap_feature]])
                        return

            if i == len(indices) - 1:
                # If we're at the end of the list, that must be the end of the last tap
                if data[tap_feature][ indices[i] ] == 1:
                    data[tap_feature].loc[ indices[i] ] = 3
                elif indices[i] + 1 < data.shape[0]:
                    data[tap_feature].loc[ indices[i] + 1] = 3
                else:
                    data[tap_feature].loc[ indices[i] ] = 0 # Remove the miscreant
                    print("There's an issue with a tap at the very last point of the data...")

        if sum(data[data[tap_feature] == 2][tap_feature]) * 3 != sum(data[data[tap_feature] == 3][tap_feature]) * 2:
            print("Uh oh, we placed an unbalanced number of 2's and 3's. Thanos would be disappointed.")
        

        # Step 2: Put a 4 at the start of the "before" window
        # and a 5 at the end of the "after" window

        start_indices = data[data[tap_feature] == 2].index
        end_indices = data[data[tap_feature] == 3].index
        if len(start_indices) != len(end_indices):
            print("Impossible.")

        #We should be able to get a half_delta on either side of
        #each window
        half_delta = delta // 2


        for i in range(len(start_indices)):
            find_index_before = start_indices[i]
            range_min = data[time_col][ start_indices[i] ] - half_delta
            while find_index_before > 0 and data[time_col][find_index_before] > range_min \
                  and data[tap_feature][find_index_before - 1] < 2:
                find_index_before -= 1
            if data[tap_feature][find_index_before] == 0:
                data[tap_feature].loc[find_index_before] = 4
            elif data[tap_feature][find_index_before] == 5 and data[tap_feature][find_index_before + 1] == 0:
                # Keep our windows from overlapping - don't put the start of one on
                # top of the end of the previous
                data[tap_feature].loc[find_index_before + 1] = 4
            elif find_index_after == 0 and data[tap_feature][find_index_after + 1] == 0:
                # If we're at the start of the interval, shift what was there forward one
                data[tap_feature].loc[find_index_after + 1] = data[tap_feature].loc[find_index_after]
                data[tap_feature].loc[find_index_after] = 4
            elif find_index_before == start_indices[i] and data[tap_feature][find_index_before - 1] == 5 \
                 and find_index_before >= 2 and data[tap_feature][find_index_before - 2] < 2:
                data[tap_feature].loc[find_index_before - 2] = 5
                data[tap_feature].loc[find_index_before - 1] = 4
            else:
                # The most likely case is that we hit the beginning or end of the
                # interval, in which case we should probably just throw the point out
                print("Oh no, that's pretty weird: ", data[tap_feature][find_index_before], find_index_before, start_indices[i])
                

            find_index_after = end_indices[i]
            range_max = data[time_col][ end_indices[i] ] + half_delta
            while find_index_after + 1 < data.shape[0] and data[time_col][find_index_after] < range_max \
                  and data[tap_feature][find_index_after + 1] < 2:
                find_index_after += 1
            if data[tap_feature][find_index_after] == 0:
                data[tap_feature].loc[find_index_after] = 5
            elif find_index_after == data.shape[0] - 1 and data[tap_feature][find_index_after - 1] == 0:
                # If we're at the end of the interval, shift what was there back one
                data[tap_feature].loc[find_index_after - 1] = data[tap_feature].loc[find_index_after]
                data[tap_feature].loc[find_index_after] = 5
            elif find_index_after == end_indices[i] and data[tap_feature][find_index_after + 1] < 2:
                data[tap_feature].loc[find_index_before + 1] = 5
            else:
                # See above comment
                print("Oh no, that's REALLY weird", find_index_after, data[tap_feature])
            
def get_feature_names():
    return ['UserID', 'SessionID', 'TaskName', 'Orientation', 'TapType'] + get_numerical_feature_names()

def get_numerical_feature_names():
    names = []
    hmog_feature_names = lambda x: [x + '_mean_during', x + '_sd_during', x + '_difference_before_after',
                                    x + '_net_change_due_to_tap', x + '_max_change', x + '_restoration_time',
                                    x + '_normalized_duration', x + '_normalized_duration_max']
    for file_name in file_names:
        for y in y_columns[file_name]:
            names += hmog_feature_names(y)
    return names

def log_column(data: DataFrame, column: str):
    """
    Take the natural log of some of the bigger columns
    """
    return data[column].map(lambda x: np.log(np.absolute(x)))

def normalize_column(data: DataFrame, column: str):
    """
    Find the Z score equivalent of each column
    """
    m = mean(data[column])
    s = sd(data[column])
    return data[column].map(lambda x: (x - m) / s)

def feature_list(user_id: str, session: str, tap_feature: str, task_name: str, window: DataFrame):
    """
    A list of features identifying a tap
    :param window: a DataFrame with just the data needed for the computation
    """
    #Add user ID, session, task name
    features = [user_id, session, task_name]

    #Add orientation
    orientation = mode(window['Phone_orientation_accel'])
    features.append(orientation)

    #Add tap type
    features.append(tap_feature)

    lead_file = 'Accelerometer.csv'

    time_col = x_columns[lead_file]

    before_start = window[window[tap_feature] == 4].index[0]
    during_start = window[window[tap_feature] == 2].index[0]
    after_start = window[window[tap_feature] == 3].index[0] + 1
    after_end = window[window[tap_feature] == 5].index[0]

    before = window.loc[before_start : during_start]
    during = window.loc[during_start : after_start]
    after = window.loc[after_start : after_end + 1]

    if during.shape[0] < 2:
        # If there were none or one measurements during the tap,
        # add the closest ones
        during = window[during_start - 1 : after_start + 1]

    for file_name in file_names:
        for y in y_columns[file_name]:

            # Feature 1: Mean during
            mean_during = mean(during[y])

            # Feature 2: SD during
            sd_during = sd(during[y])

            # Feature 3: Difference before/after
            mean_before = mean(before[y])
            mean_after = mean(after[y])
            difference_before_after = mean_after - mean_before

            # Feature 4: Net change from tap
            net_change_due_to_tap = mean_during - mean_before

            # Feature 5: Maximal change from tap
            max_tap = max(during[y])
            max_change = max_tap - mean_before

            # Feature 6: Restoration time
            avgDiffs = []
            for j in range(after[y].shape[0]):
                subsequentValues = after[y].iloc[j:]
                subsequentDistances = subsequentValues.map(lambda x: abs(x - mean_before))
                averageDistance = mean(subsequentDistances)
                avgDiffs.append(averageDistance)
            time_of_earliest_restoration = min(avgDiffs)
            restoration_time = time_of_earliest_restoration - during[time_col].iloc[-1]

            # Feature 7: Normalized duration
            t_before_center = (before[time_col].iloc[0] + before[time_col].iloc[-1]) / 2 
            t_after_center = (after[time_col].iloc[0] + after[time_col].iloc[-1]) / 2
            normalized_duration = (t_after_center - t_before_center) / (mean_after - mean_before)
            
            # Feature 8: Ndormalized duration max
            t_max_in_tap = during[during[y] == max_tap][time_col].iloc[0]
            normalized_duration_max = (t_after_center - t_max_in_tap) / (mean_after - max_tap)


            features += [mean_during, sd_during, difference_before_after,
                                net_change_due_to_tap, max_change, restoration_time,
                                normalized_duration, normalized_duration_max]
    
    return features

def get_feature_vector(user_id: str, session: str) -> DataFrame:
    """
    This will create a list of motion-related features characterizing each tap of
    the given user and session
    """

    #Find the time windows during which the reader is doing the desired task
    activity_data = read_file(user_id, session, 'Activity.csv')
    task_number = mode(activity_data['TaskID'])
    task_name = task_names[(task_number - 1) % len(task_names)]

    tap_windows = get_tap_events(user_id, session)
    data = get_user_session_data(user_id, session)
    add_magnitude_columns(data)
    add_columns_for_taps(data, tap_windows)
    mark_tap_start_and_end(data, delta_in_ms = 200)

    column_names = get_feature_names()

    #A feature vector for each tap, to be filled in subsequently:
    featureVectors = pd.DataFrame(columns = column_names)

    for tap_file in tap_file_names:
        tap_feature = tap_file_to_feature_name[tap_file]
        print(tap_feature)
        window_start_indices = data[data[tap_feature] == 4].index
        window_end_indices = data[data[tap_feature] == 5].index
        if len(window_start_indices) == 0:
            continue
        
        for i in range(len(window_start_indices)):
            start, end = window_start_indices[i], window_end_indices[i]
            window_of_interest = data[start : end + 1]
            featureVectors.loc[featureVectors.shape[0]] = feature_list(user_id, session, tap_feature, task_name, window_of_interest)
    
    return featureVectors

                

                # We will make a few plots as we go
                # if (y == 'gyroY' and random.choice(range(100)) == 1): #before[0] == 2214882326000):
                #     print("Making a plot at time " + str(before[0]))
                #     beforePoints, = plt.plot(before, before_values, 'bo', color = 'blue', label = 'Before Tap')
                #     afterPoints, = plt.plot(after, after_values, 'bo', color = 'red', label = 'After Tap')
                #     duringPoints, = plt.plot(during, during_values, 'bo', color = 'black', label = 'During Tap')
                #     plt.xlabel('Event Time')
                #     plt.ylabel(y)

                #     plt.legend(handles = [beforePoints, duringPoints, afterPoints])

                    # Uncomment this to get some key values marked on the plot:

                    # min_x = before[0] - (before[1] - before[0]) * 10
                    # min_y = min([min(during_values), min(before_values), min(after_values)])
                    # # Mark the mean during tap event (Feature 1)
                    # plt.hlines(y = mean_during, xmin = min_x, xmax = during[-1], linestyle='dashed', \
                    #             color='black')
                    # plt.annotate(xy = (min_x, mean_during), s = 'avgDuringTap')
                    # # Mark the mean before
                    # plt.hlines(y = mean_before, xmin = min_x, xmax = before[-1], linestyle='dashed', \
                    #             color='blue')
                    # plt.annotate(xy = (min_x, mean_before), s = 'avg100msBefore')
                    # # Mark the mean after
                    # plt.hlines(y = mean_after, xmin = min_x, xmax = after[-1], linestyle='dashed', \
                    #             color='red')
                    # plt.annotate(xy = (min_x, mean_after), s = 'avg100msAfter')
                    # # Mark the max in the tap
                    # plt.hlines(y = max_tap, xmin = min_x, xmax = t_max_in_tap, linestyle='dashed', \
                    #             color='black')
                    # plt.annotate(xy = (min_x, max_tap), s = 'maxTap')

                    # # Mark the important points
                    # plt.vlines(x = t_before_center, ymin = min_y, ymax = mean_before, linestyle='dashed', \
                    #             color='blue')
                    # plt.annotate(xy = (t_before_center, min_y), s = 'tBeforeCenter')
                    # plt.vlines(x = t_after_center, ymin = min_y, ymax = mean_after, linestyle='dashed', \
                    #             color='red')
                    # plt.annotate(xy = (t_after_center, min_y), s = 'tAfterCenter')
                    # plt.vlines(x = t_max_in_tap, ymin = min_y, ymax = max_tap, linestyle='dashed', \
                    #             color='black')
                    # plt.annotate(xy = (t_max_in_tap, min_y), s = 'tMax')
                    # # plt.vlines(x = time_of_earliest_restoration, ymin = min_y, ymax = mean_before, linestyle='dashed', \
                    # #             color = 'red')
                    # # plt.annotate(xy = (time_of_earliest_restoration, min_y), s = 'tRestoration')

                    # plt.savefig('Plots/Project/'+session+'_'+y+'_time_'+str(before[0]))

                    # plt.close()

    return features

def get_distance(user_id1: str, user_id2: str) -> float:
    """
    Uses some metric TBD to evaluate a distance between the two users' feature vectors
    """
    features1 = get_feature_vector(user_id1)
    features2 = get_feature_vector(user_id2)
    pass

def get_clusters() -> List[List[str]]:
    """
    Uses a clustering algorithm TBD (K means?) and the get_distance function to group the
    users from the data set into clusters
    """
    all_users = get_user_ids()
    pass

def write_to_csv(list_of_rows, file_name):
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        for row in list_of_rows:
            if None in row:
                continue
            writer.writerow(row)
        
    f.close()

def write_all_users(folder_name: str, label: bool):
    """
    Writes a .csv file of features for each user in the given location.
    If label is True, the first row of the files will be a header.
    """
    make_directory(folder_name)
    for user in get_user_ids():
        print("Analysis of user: " + user)
        subfolder_name = folder_name + "/" + user
        make_directory(subfolder_name)
        for session in get_user_session_ids(user):
            print("Session: " + session)
            file_name = subfolder_name + "/" + session + ".csv"
            data = get_feature_vector(user, session)
            if data == None:
                continue
            if label:
                data = [labels] + data
            write_to_csv(data, file_name)


def write_random_user():
    """
    Writes a .csv file of features for each user in the given location.
    If label is True, the first row of the files will be a header.
    """
    folder_name = 'TapData'
    make_directory(folder_name)
    label = False
    user = random.choice(get_user_ids())
    print("Analysis of user: " + user)
    subfolder_name = folder_name + "/" + user
    make_directory(subfolder_name)
    session = random.choice(get_user_session_ids(user))
    print("Session: " + session)
    file_name = subfolder_name + "/" + session + ".csv"
    data = get_feature_vector(user, session)
    if data != None and label:
        data = [labels] + data
    write_to_csv(data, file_name)

def find_identifying_shapes(user_id: str, session: str, plot_histograms = True, return_shape_vectors = True,
                            path = 'Plots/Research/', tap_feature_name = None, reading_value = None):
    """
    Create 2d histogram plots of the "average touch event" for each kind of touch event,
    and write them to the "Plots/Research" folder
    We're going to return vectors representing the shape of the 2d histogram plot of the form:
    ( [List of vectors] , [List of corresponding labels] )
    """
    make_directory(path)
    folder_name = path+user_id
    make_directory(folder_name)
    
    activity_data = read_file(user_id, session, 'Activity.csv', \
                            ['Relative_Start_time', 'Relative_End_time', 'TaskID'])
    
    data = get_user_session_data(user_id, session)

    shapeVectors = []
    labels = []

    for task_name in task_names:
        # We're going to make a plot for each particular task
        windows_to_consider = []

        num_rows = activity_data.shape[0]
        for row in range(num_rows):
            start = activity_data.iloc[row, 0]
            end = activity_data.iloc[row, 1]
            task_number = activity_data.iloc[row, 2]
            task = task_names[(task_number - 1) % len(task_names)]
            if (task == task_name):
                windows_to_consider.append([start, end])

        if windows_to_consider == []:
            continue

        subfolder_name = folder_name + '/' + task_name
        make_directory(subfolder_name)
        subfolder_name += '/' + session
        make_directory(subfolder_name)

        tap_windows = {}
        for tap_file in tap_file_names:
            tap_windows[tap_file] = get_tap_events(user_id, session, windows_to_consider, tap_file)

        # Compute the relevant time values for each file (accelerometer and gyroscope)
        time_values_for_each_file = {}
        data_xy_for_each_file = {}
        for file_name in file_names:
            x = x_columns[file_name]
            time_values_for_each_file[file_name] = {}
            for tap_file in tap_file_names:
                time_values_for_each_file[file_name][tap_file] = \
                    time_values_during_taps(user_id, session, data, x, tap_windows[tap_file])
            data_xy_for_each_file[file_name] = read_file(user_id, session, file_name, \
                [x] + y_columns[file_name] + other_columns[file_name])

        min_num_taps = {}

        for tap_file in tap_file_names:

            min_num_taps[tap_file] = len(tap_windows[tap_file])

            for file_name in file_names:
                num_taps = len(time_values_for_each_file[file_name][tap_file])
                if min_num_taps[tap_file] > num_taps:
                    min_num_taps[tap_file] = num_taps

        pointWindow = 10 # usually 10

        for tap_file in tap_file_names:
            if tap_feature_name != None and tap_file != tap_feature_to_file_name[tap_feature_name]:
                continue

            pointsNearTapStart = {}
            # cubicCoefficients = {}

            for file_name in file_names:
                for y in y_columns[file_name]:
                    # This assumes every name in y_columns is distinct;
                    # you may get an error if they are not
                    pointsNearTapStart[y] = []

                    # cubicCoefficients[y] = [[], [], []]
                        
            subsubfolder_name = subfolder_name + '/' + tap_file_to_feature_name[tap_file]
            make_directory(subsubfolder_name)

            for window_index in range(min_num_taps[tap_file]):
                for file_name in file_names:
                    x = x_columns[file_name]
                    time_values = time_values_for_each_file[file_name][tap_file]
                    data_xy = data_xy_for_each_file[file_name]

                    # Here we look at one particular tap, given by window
                    window = time_values[window_index]
                    before = window[0]

                    if len(before) == 0:
                        continue

                    x_values = list(data_xy.loc[:, x])
                    start_index = x_values.index(before[-1]) + 1
                    time_zero = (data_xy[x][start_index - 1] + data_xy[x][start_index]) / 2
                    
                    if start_index - pointWindow < 0 or start_index + pointWindow > data_xy.shape[0]:
                        continue

                    for y in y_columns[file_name]:
                        if reading_value != None and y != reading_value:
                            continue
                        # Keep track of points to add to scatter
                        value_zero = (data_xy[y][start_index - 1] + data_xy[y][start_index]) / 2
                        for j in range(start_index - pointWindow, start_index + pointWindow):
                            pointsNearTapStart[y].append( (data_xy[x][j] - time_zero, data_xy[y][j] - value_zero) )

                        # # Keep track of fit coefficients
                        # cubicFit = coefficients(3, data_xy, x, y, start_index, pointWindow // 2)[:-1]
                        # # Exclude the last coefficient because it is a constant term and does not affect shape
                        # for i in range(len(cubicCoefficients[y])):
                        #     cubicCoefficients[y][i].append(cubicFit[i])

            # Now that we have filled up ten values for each tap event,
            # we must make the plots

            for file_name in file_names:
                for y in y_columns[file_name]:
                    if reading_value != None and y != reading_value:
                            continue
                    if len(pointsNearTapStart[y]) == 0:
                        break
                    xCoords, yCoords = extract_coordinates(pointsNearTapStart[y])
                    plot_range = [[MIN_X,MAX_X], [MIN_Y,MAX_Y]]#percentile_region(xCoords, yCoords, percent = 90)

                    # scatterplot(xCoords, yCoords, range = plot_range, file_name = subsubfolder_name + '/CombinedScatter_'+y)
                    
                    if plot_histograms:
                        histogram2d(xCoords, yCoords, plot_range, file_name = subsubfolder_name + '/HullPlot_'+y)

                    if return_shape_vectors:
                        shapeVectors.append(shape_vector(xCoords, yCoords, [[MIN_X, MAX_X], [MIN_Y, MAX_Y]], BIN_NUMBER))
                        labels.append( headingName(user_id, session, task_name, tap_file_to_feature_name[tap_file], y) )

                    # scatterplot3D(cubicCoefficients[y][0], cubicCoefficients[y][1], \
                    #     cubicCoefficients[y][2], file_name = subsubfolder_name + '/CoefficientsPlot_'+y)
        return shapeVectors, labels

def headingName(user, session, task_name, tap_feature_name, reading_value):
    return task_name + ", " + reading_value + ", " + tap_feature_name + ", " + session

def coefficients(degree: int, data: DataFrame, x_col: str, y_col: str, index: int, radius: int) -> List[float]:
    """
    Computes polynomial fit coefficients for interval [index - radius, index + radius)
    :param degree: degree of fit
    :param data: DataFrame containing the columns we're using for x and y
    :param x_col: name of the x column within data
    :param y_col: name of the y column within data
    :param index: index within data to center the fit at
    :param radius: width of interval on each side of index (inclusive to left, not right)
    """
    x_values = data.loc[index - radius : index + radius, x_col]
    y_values = data.loc[index - radius : index + radius, y_col]
    x_center = data[x_col][index]

    x_values = [element - x_center for element in x_values]

    fit = list(np.polyfit(x_values, y_values, degree))

    return fit

def extract_coordinates(list_of_points: List[List[float]]):
    """
    Given a list of ordered pairs representing points in the xy-plane,
    returns separate lists for the x coordinates and y coordinates
    :param list_of_points: In the form [[x1, y1], [x2, y2], [x3, y3], ...]
    :return The same points, but in the form ([x1, x2, x3, ...], [y1, y2, y3, ...])
    """
    xCoords = [point[0] for point in list_of_points]
    yCoords = [point[1] for point in list_of_points]
    return (xCoords, yCoords)

def percentile_region(xCoords, yCoords, percent = 90) -> List[List[float]]:
    """
    Returns the boundaries of a rectangular region containing
    some percent of the x and y values
    :param xCoords: array-like, x-coordinates of the points
    :param yCoords: array-like, y-coordinates of the points
    :param percent: percent of the values in xCoords will be between the x limits,
                    and percent of those in yCoords will be between the y limits
    :return the boundaries of the rectangular region, in the form
        [[xMin, xMax], [yMin, yMax]] 
    """
    excluded = (100 - percent) / 2
    xMin = stat.scoreatpercentile(xCoords, per = excluded)
    xMax = stat.scoreatpercentile(xCoords, per = 100 - excluded)
    yMin = stat.scoreatpercentile(yCoords, per = excluded)
    yMax = stat.scoreatpercentile(yCoords, per = 100 - excluded)
    return [[xMin, xMax], [yMin, yMax]]

def scatterplot(xCoords, yCoords, range, file_name = None):
    """
    Plots a scatterplot in 3 dimensions
    :param xCoords: array-like, x-coordinates of the points
    :param yCoords: array-like, y-coordinates of the points
    :param range: Of the form [[xMin, xMax],[yMin,yMax]]
    :param file_name: file to save to, will just show if none is provided
    """
    plt.plot(xCoords, yCoords, range = range)
    if file_name == None:
        plt.show()
    else:
        plt.savefig(file_name)
    plt.close()

def histogram2d(xCoords, yCoords, range, num_bins = 40, file_name = None):
    """
    Plots a histogram for the given points within the specified range
    :param xCoords: array-like, x-coordinates of the points
    :param yCoords: array-like, y-coordinates of the points
    :param range: Of the form [[xMin, xMax],[yMin,yMax]]
    :param num_bins: Number of boxes to subdivide each dimension into, default 40
    :param file_name: file to save to, will just show if none is provided
    """
    plt.hist2d(xCoords, yCoords, num_bins, range = range)
    if file_name == None:
        plt.show()
    else:
        plt.savefig(file_name)
    plt.close()

def scatterplot3d(xCoords, yCoords, zCoords, file_name = None):
    """
    Plots a scatterplot in 3 dimensions
    :param xCoords: array-like, x-coordinates of the points
    :param yCoords: array-like, y-coordinates of the points
    :param zCoords: array-like, z-coordinates of the points
    :param file_name: file to save to, will just show if none is provided
    """
    ax = plt.axes(projection = '3d')
    ax.scatter3D(xCoords, yCoords, zCoords)
    if file_name == None:
        plt.show()
    else:
        plt.savefig(file_name)
    plt.close()

def make_directory(name: str):
    """
    Will make a directory with the given name, unless
    such a directory already exists, in which case nothing
    will happen
    :param name: the name of the directory
    """
    try:
        os.mkdir(name)
    except:
        pass

def shape_vector(xCoords, yCoords, plot_range, num_bins = 40):
    """
    This is a vector representing the shape of the 2d histogram
    representing the average of a person's tap data
    """
    xMin, xMax = plot_range[0][0], plot_range[0][1]
    yMin, yMax = plot_range[1][0], plot_range[1][1]
    counts = [0 for i in range(num_bins ** 2)]
    total = 0
    binWidth_x = (xMax - xMin) / num_bins
    binWidth_y = (yMax - yMin) / num_bins
    for i in range(len(xCoords)):
        x, y = xCoords[i], yCoords[i]
        x_num = int( (x - xMin) / binWidth_x )
        y_num = int( (y - yMin) / binWidth_y )
        if (x_num >= 0 and x_num < num_bins and y_num >= 0 and y_num < num_bins):
            counts[x_num * num_bins + y_num] += 1
            total += 1
    if total > 0:
        for i in range(num_bins):
            for j in range(num_bins):
                counts[i * num_bins + j] /= total
        # New feature: total number of points at the end
    counts.append(total)
    return counts

def write_shape_csv_for_each_user(task_name, tap_feature_name, reading_value,
                                  list_of_users = None, histograms = True, path = 'Plots/Research/'):
    make_directory(path)
    if list_of_users == None:
        list_of_users = get_user_ids()
    for user in list_of_users:
        print("Investigating user: " + user)
        make_directory(path+user)
        with open(path+user+'/ShapeVectors.csv', mode = 'w') as file:
            writer = csv.writer(file)
            allLabels = []
            allVectors = []
            for session in get_user_session_ids_for_task(user, task_name):     
                print("Session: " + session)
                shape_vector, labels = find_identifying_shapes(user, session, plot_histograms = histograms, \
                                                               return_shape_vectors = True, path = path, \
                                                               tap_feature_name = tap_feature_name, reading_value = reading_value)
                allLabels += labels
                allVectors.append(shape_vector)
            writer.writerow(allLabels)
            for i in range(BIN_NUMBER ** 2):
                rowToWrite = []
                for s in range(len(allVectors)):
                    for v in range(len(allVectors[s])):
                        rowToWrite.append(allVectors[s][v][i])
                writer.writerow(rowToWrite)
        file.close()
    print("Finished!")

def get_csv_column(heading: str, filename: str):
    data = pd.read_csv(filename)
    return data[heading]

def distance(user1: str, session1: str, user2: str, session2: str, task_name: str, tap_feature_name: str, reading_value: str):
    # First we get the vectors
    heading1 = headingName(user1, session1, task_name, tap_feature_name, reading_value)
    heading2 = headingName(user2, session2, task_name, tap_feature_name, reading_value)
    filename1 = 'Plots/Research/' + user1 + '/ShapeVectors.csv'
    filename2 = 'Plots/Research/' + user2 + '/ShapeVectors.csv'
    shape_vector1 = get_csv_column(heading1, filename1)
    shape_vector2 = get_csv_column(heading2, filename2)

    # Then we the Euclidean norm of the difference in the vectors
    sum = 0
    for index, value in shape_vector1.iteritems():
        sum += (shape_vector1[index] - shape_vector2[index]) ** 2
    return np.sqrt(sum)





# Write shape vector csvs:
# write_shape_csv_for_each_user()


# Make 2d histograms:
# str1 = ""
# for user in get_user_ids():

#     if int(user) < 733162:
#         str1 += user + ", "
#         continue
#     # else:
#     #     print(str1)
#     #     break
#     print("Investigating user: " + user)          
#     for session in get_user_session_ids(user):      
#         print("Session: " + session)
#         find_identifying_shapes(user, session, plot_histograms = True, return_shape_vectors = False)

# def get_num_events(user_id, session, task_name, tap_file):
#     folder_name = 'Plots/Research/'+user_id
#     make_directory(folder_name)
    
#     activity_data = read_file(user_id, session, 'Activity.csv')
#     task_number = mode(activity_data['TaskID'])
#     task_name = task_names[(task_number - 1) % len(task_names)]
    
#     data = get_user_session_data(user_id, session)

#     shapeVectors = []
#     labels = []

#     subfolder_name = folder_name + '/' + task_name
#     make_directory(subfolder_name)
#     subfolder_name += '/' + session
#     make_directory(subfolder_name)

#     tap_windows = get_tap_events(user_id, session, windows_to_consider, tap_file)   
#     return len(tap_windows)

def get_num_events_for_user(user, task_name, tap_file):
    return [get_num_events(user, session, task_name, tap_file) \
        for session in get_user_session_ids_for_task(user, task_name)]

def above_threshold(user, session, task_name, tap_file):
    return get_num_events(user, session, task_name, tap_file) >= THRESHOLD

def find_distances(task_name, tap_feature_name, reading_value, list_of_users = None):
    if (list_of_users == None):
        list_of_users = list(filter(lambda x: int(x) != 733162, get_user_ids())) #733162 has incomplete data
    tap_file_name = tap_feature_to_file_name[tap_feature_name]
    same_user = []
    different_user = []
    for u1 in list_of_users:
        for session1 in get_user_session_ids_for_task(u1, task_name):
            with open('Plots/Research/'+u1+'/'+task_name+'/'+session1+'/'+tap_feature_name+'/Distances.csv', mode = 'w') as file:
                writer = csv.writer(file)
                print("Comparing " + session1 + " to other sessions")
                # if (not above_threshold(u1, session1, task_name, tap_file_name)):
                #     continue
                for u2 in list_of_users:
                    for session2 in get_user_session_ids_for_task(u2, task_name):
                        if session1 >= session2:
                            continue # Only investigate each pair once
                        #print(session2)
                        # if (not above_threshold(u2, session2, task_name, tap_file_name)):
                        #     continue
                        d = distance(u1, session1, u2, session2, task_name, tap_feature_name, reading_value)
                        if u1 == u2:
                            same_user.append(d)
                        else:
                            different_user.append(d)
                        # print('Distance between ' + session1 + ' and ' + session2 + ' is:')
                        # print(d)
                        writer.writerow([session2, d])
            file.close()
    print("Average for the same user:")
    print(mean(same_user))
    print("Standard deviation for the same user:")
    print(sd(same_user))
    print("Average for different users:")
    print(mean(different_user))
    print("Standard deviation for different users:")
    print(sd(different_user))

def get_distance_already_computed(user1, session1, user2, session2, task_name, tap_feature_name, reading_value):
    tap_file_name = tap_feature_to_file_name[tap_feature_name]
    userA = min(user1, user2)
    sessionA = min(session1, session2)
    userB = max(user1, user2)
    sessionB = max(session1, session2)
    with open('Plots/Research/'+userA+'/'+task_name+'/'+sessionA+'/'+tap_feature_name+'/Distances.csv', mode = 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == sessionB:
                return float(row[1])
    return 100

def test_user_identification(task_name, tap_feature_name, reading_value, list_of_users):
    training_sessions = {}
    testing_sessions = {}
    for user in list_of_users:
        sessions = get_user_session_ids_for_task(user, task_name)
        training_sessions[user] = sessions[0:2]
        testing_sessions[user] = sessions[2:4]
    num_successes = 0
    num_attempts = 0
    misclassifications = []
    for user1 in list_of_users:  
        for session1 in testing_sessions[user1]:
            min_distance = 100
            min_user = 'NONE'
            min_session = 'None'
            for user2 in list_of_users:
                for session2 in training_sessions[user2]:
                    dist = get_distance_already_computed(user1, session1, user2, session2, task_name, tap_feature_name, reading_value)
                    if dist < min_distance:
                        min_distance = dist
                        min_user = user2
                        min_session = session2
            if min_user == user1:
                num_successes += 1
            else:
                misclassifications.append([session1, session2])
            num_attempts += 1
    print("Number of attempted identifications: " + str(num_attempts))
    print("Number of successful identifications: " + str(num_successes))
    print("Success rate: " + str(num_successes / num_attempts))
    if (len(misclassifications) > 0):
        print("Unsuccessful classifications:")
        for mis in misclassifications:
            print("The nearest neighbor of " + mis[0] + " was " + mis[1])


# This evaluates some distances:

# task_name = 'Writing+Sitting'
# tap_feature_name = 'KeyPress'
# reading_value = 'accelZ'

# all_users_with_data = list(filter(lambda x: int(x) != 733162, get_user_ids()))
# training_sessions = {}
# testing_sessions = {}
# for user in all_users_with_data:
#     sessions = get_user_session_ids_for_task(user, task_name)
#     training_sessions[user] = sessions[0:2]
#     testing_sessions[user] = sessions[2:4]
# num_successes = 0
# num_attempts = 0
# for user1 in all_users_with_data:  
#     for session1 in testing_sessions[user1]:
#         min_distance = 100
#         min_user = 'NONE'
#         for user2 in all_users_with_data:
#             for session2 in training_sessions[user2]:
#                 dist = get_distance_already_computed(user1, session1, user2, session2, task_name, tap_feature_name, reading_value)
#                 if dist < min_distance:
#                     min_distance = dist
#                     min_user = user2
#         if min_user == user1:
#             num_successes += 1
#         else:
#             print("Attempting to classify " + session1)
#             print("It was most similar to a session from " + min_user)
#         num_attempts += 1
# print("Number of attempts: ")
# print(num_attempts)
# print("Number of successes: ")
# print(num_successes)



# for task_name in task_names:
#     print(task_name)
#     for tap_file_name in tap_file_names:
#         print(tap_file_name)
#         for file_name in file_names:
#             print(file_name)
#             for reading_value in y_columns[file_name]:
#                 if task_name == 'Writing+Sitting' and tap_feature_name == 'KeyPress' and tap_file_name == 'KeyPressEvent.csv':
#                     continue
#                 write_distances_to_files(task_name, tap_file_to_feature_name[tap_file_name], reading_value)

# user = random.choice(get_user_ids())
# session = random.choice(get_user_session_ids(user))
# data = pd.read_csv('AllTaps/'+user+'/'+session+'.csv')[1:]
# numeric_data = data[data.columns[4:52]]

# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# pca.fit(numeric_data)
# print(pca.components_)
# print(pca.explained_variance_)

# comp1 = pca.components_[0]
# comp2 = pca.components_[1]

# for i in range(numeric_data.shape[0]):
#     vec = numeric_data.iloc[i]
#     xComponent = np.dot(vec, comp1)
#     yComponent = np.dot(vec, comp2)
#     plt.plot(xComponent, yComponent, 'bo', color = 'blue' )
# plt.show()
# plt.close()

# from sklearn.manifold import Isomap
# isomap = Isomap(n_neighbors=5, n_components=2)
# isomap.fit(numeric_data)
# manifold_2Da = isomap.transform(numeric_data)
# manifold_2D = pd.DataFrame(manifold_2Da, columns=['Component 1', 'Component 2'])

# plt.plot(manifold_2D['Component 1'], manifold_2D['Component 2'], 'bo', color = 'blue')
# plt.show()

# users = 10
# sessions_for_each = 2
# make_directory('NewData')
# for user_id in random.sample(get_user_ids(), users):
#     print(user_id)
#     frames = []
#     for session in random.sample(get_user_session_ids(user_id), sessions_for_each):
#         print(session)
#         try:
#             frames.append(get_feature_vector(user_id, session))
#         except:
#             print("Error in session " + session)
#             continue
#     features = pd.concat(frames, ignore_index = True)
# features.to_csv('NewData/RandomSample.csv', header = True, index = False)

# from sklearn.manifold import Isomap
# isomap = Isomap(n_neighbors=5, n_components=2)
# isomap.fit(numeric_data)
# manifold_2Da = isomap.transform(numeric_data)
# manifold_2D = pd.DataFrame(manifold_2Da, columns=['Component 1', 'Component 2'])

# plt.plot(manifold_2D['Component 1'], manifold_2D['Component 2'], 'bo', color = 'blue')
# plt.show()
