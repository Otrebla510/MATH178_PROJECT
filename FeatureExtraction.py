
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
    'Activity.csv': ['ID', 'SubjectID', 'Session_number', 'Start_time', 'End_time', 'Relative_Start_time', 'Relative_End_time',
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

def add_columns_for_taps(full_data: DataFrame, tap_data: DataFrame):
    """
    This is a step of the processing of the data from the accelerometer/gyroscope. It
    adds a new column for each kind of tap, indicating whether a tap of that kind is
    in progress.
    :param full_data: The combined dataframe from the accelerometer/gyroscope (returned by get_user_session_data)
    :param tap_data: The dataframe indicating the start and end of each tap event (returned by get_tap_events)
    """
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
    """
    The names of the features, or the columns of the csv file that will be written
    """
    return ['UserID', 'SessionID', 'TaskName', 'Orientation', 'TapType'] + get_numerical_feature_names()

def get_numerical_feature_names():
    """
    The names of the HMOG features
    """
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
    Take the natural log of the specified column
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
    if window.shape[0] == 0:
        return None
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

    if random.choice(range(100))== 0:
        plot_tap('Plots/Project/' + session, before, during, after, time_col)
    
    return features


def plot_tap(file: str, before: DataFrame, during: DataFrame, after: DataFrame, time_col: str):
    """
    Make plots for each reading showing the times series during the 
    tap, and write them to the Plots folder. The plots will have the mean
    within each time segment marked.
    :param file: The base file name (each output file will be named file + [an extension
                 indicating which reading they come from])
    :param before,during,after: The data in three time segments
    :param time_col: The name of the column of each dataframe being used as 'time'
    """

    print("Making plots at time " + str(before[time_col].iloc[0]))

    for file_name in file_names:
        for y in y_columns[file_name]:

            ax = before.plot(time_col, y, kind = 'scatter', color = 'blue', label = 'Before Tap')
            after.plot(time_col, y, kind = 'scatter', color = 'red', label = 'After Tap', ax = ax)
            during.plot(time_col, y, kind = 'scatter', color = 'black', label = 'During Tap', ax = ax)
            plt.axes(ax)
            plt.xlabel('Event Time')
            plt.ylabel(y)

            min_x = before[time_col].iloc[0] - (before[time_col].iloc[1] - before[time_col].iloc[0]) * 50
            min_y = min([min(during[y]), min(before[y]), min(after[y])])
            # Mark the mean during tap event (Feature 1)
            mean_during = mean(during[y])
            mean_before = mean(before[y])
            mean_after = mean(after[y])
            plt.hlines(y = mean_during, xmin = min_x, xmax = during[time_col].iloc[-1], linestyle='dashed', \
                        color='black')
            plt.annotate(xy = (min_x, mean_during), s = 'avgDuringTap')
            # Mark the mean before
            plt.hlines(y = mean_before, xmin = min_x, xmax = before[time_col].iloc[-1], linestyle='dashed', \
                        color='blue')
            plt.annotate(xy = (min_x, mean_before), s = 'avg100msBefore')
            # Mark the mean after
            plt.hlines(y = mean_after, xmin = min_x, xmax = after[time_col].iloc[-1], linestyle='dashed', \
                        color='red')
            plt.annotate(xy = (min_x, mean_after), s = 'avg100msAfter')

            plt.legend()

            plt.savefig(file+'_'+y+'_time_'+str(before[time_col].iloc[0]) + '.png')

            plt.close()

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
            features = feature_list(user_id, session, tap_feature, task_name, window_of_interest)
            if features != None:
                featureVectors.loc[featureVectors.shape[0]] = features
    
    return featureVectors

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
    """
    A helper function for writing a list of rows to a DataFrame
    """
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