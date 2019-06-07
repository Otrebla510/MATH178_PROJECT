
from pathlib import Path
from typing import List
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from statistics import mode
from statistics import mean
from statistics import stdev as sd

import os
import csv
import random
import numpy as np

# Data pre processing
base_data_folder_path = Path('public_dataset')
file_name_to_column_names = {
    'Accelerometer.csv': ['Systime_accel', 'EventTime_accel', 'ActivityID', 'accelX', 'accelY', 'accelZ', 'Phone_orientation_accel'],
    'Activity.csv': ['ID', 'SubjectID', 'Start_time', 'End_time', 'Relative_Start_time', 'Relative_End_time',
                     'Gesture_scenario', 'TaskID', 'ContentID'],
    'Gyroscope.csv': ['Systime_gyro', 'EventTime_gyro', 'ActivityID', 'gyroX', 'gyroY', 'gyroZ', 'Phone_orientation_gyro'],
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
tap_file_names = [
    'TouchEvent.csv',
    'KeyPressEvent.csv',
    'OneFingerTouchEvent.csv',
    'PinchEvent.csv',
    'ScrollEvent.csv',
    'StrokeEvent.csv']
x_columns = {
    'Accelerometer.csv': 'EventTime_accel',
    'Gyroscope.csv': 'EventTime_gyro'
}
y_columns = {
    'Accelerometer.csv': ['accelX', 'accelY', 'accelZ'],
    'Gyroscope.csv': ['gyroX', 'gyroY', 'gyroZ']
}
other_columns = {
    'Accelerometer.csv': ['Phone_orientation_accel'],
    'Gyroscope.csv': ['Phone_orientation_gyro']
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

def get_user_ids() -> List[str]:
    """
    Get all user ids based on name of folders under "public_dataset/"
    :return: a list of user ids
    """
    listOfFiles = os.listdir('public_dataset')
    listOfFiles.remove('data_description.pdf')
    listOfFiles.remove('.DS_Store')
    return listOfFiles
    
def get_user_session_ids(user_id: str) -> List[str]:
    """
    Get all session ids for a specific user based on folder structure
    e.g. "public_dataset/100669/100669_session_13" has user_id=100669, session_id=13
    :param user_id: user id
    :return: list of user session ids
    """
    listOfSessions = os.listdir('public_dataset/'+user_id)
    listOfSessions.remove('.DS_Store')
    return listOfSessions

def read_file(user_id: str, user_session_id: str, file_name: str, column_names: List[str]) -> DataFrame:
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
    data_frame = pd.read_csv(path, header=None, names = file_name_to_column_names[file_name])
    for column in full_columns:
        if not(column in column_names):
            data_frame = data_frame.drop(column, axis = 1)
    return data_frame

def get_user_session_data(user_id: str, user_session_id: str) -> DataFrame:
    """
    Combine accelerometer, gyroscope, and activity labels for a specific session of a user
    Note: Timestamps are ignored when joining accelerometer and gyroscope data.
    :param user_id: user id
    :param user_session_id: user session id
    :return: combined DataFrame for a session
    """
    df_list = []
    for i in range(0, len(file_names)):
        additional_df = read_file(user_id, user_session_id, file_names[i], file_name_to_column_names[file_names[i]])
        df_list.append(additional_df)
    df = pd.concat(df_list, axis=1, join = 'inner')
    return df

def get_tap_events(user_id: str, user_session_id: str, windows: List[List[float]]) -> List[List[float]]:
    """
    Return a list of the relative times at which tap events occur
    TODO: make it return separate lists based on phone orientation
    """
    listOfTapWindows = []
    delta = 100 #Difference for which taps can be considered separate (ms)
    for tap_file in tap_file_names:
        columns = tap_file_important_columns[tap_file]
        data = read_file(user_id, user_session_id, tap_file, columns)
        if len(columns) == 2:
            times = list(data.loc[:, data.columns.values.tolist()[0]])
            currentWindow = []
            for i in range(len(times)):
                t = int(times[i])
                if currentWindow == []:
                    currentWindow.append(t)
                else:
                    if t - currentWindow[-1] <= delta:
                        currentWindow.append(t)
                    else:
                        listOfTapWindows.append([ currentWindow[0], currentWindow[-1] ])
                        currentWindow = []
        elif len(columns) == 3:
            begin_times = list(data.loc[:, data.columns.values.tolist()[0]])
            end_times = list(data.loc[:, data.columns.values.tolist()[1]])
            for i in range(len(begin_times)):
                listOfTapWindows.append([ begin_times[i], end_times[i] ])
        else:
            print(len(columns))
            pass
    #Combine overlapping windows
    listOfTapWindows.sort(key = lambda x: x[0])
    index = 0
    while index < len(listOfTapWindows) - 1:
        window1 = listOfTapWindows[index]
        window2 = listOfTapWindows[index+1]
        if (window1[-1] >= window2[0]):
            listOfTapWindows[index] = [window1[0] , max(window1[-1], window2[-1])]
            del listOfTapWindows[index+1]
        else:
            index += 1
    return listOfTapWindows

def time_values_during_taps(user_id: str, session: str, data: DataFrame, time_column: str, \
                            tap_windows: List[List[float]]) -> List[List[List[float]]]:
    """
    Finds all time values that are within tap windows.
    Returns as [[[times 100ms before], [t1, t2, ..., tn], [times 100 ms after]], ...], with empty lists
    if no time values occur in a given tap window (unlikely, but possible)
    """
    num_rows = data.shape[0]
    delta = 100
    time_values = [ [ [],[],[] ] ]
    currentWindow = 0
    times = data[time_column].values.tolist()
    for i in range(num_rows):
                
        # Note: The data has the accelerometer/gyro times in different units,
        # so to convert the time we divide by 1000000
        unconverted_time = times[i]
        time = unconverted_time / 1000000

        if currentWindow >= len(tap_windows):
            break # We've reached every window in tap_windows

        while currentWindow >= len(time_values):
            time_values.append([ [],[],[] ]) # Start a new list for the new tap

        window = tap_windows[currentWindow]
        if time >= window[0] - delta and time < window[0]:
            time_values[currentWindow][0].append(unconverted_time) #In the 100 ms before
        elif time >= window[0] and time <= window[-1]:
            time_values[currentWindow][1].append(unconverted_time) # In the window itself
        elif time <= window[-1] + delta and time > window[-1]:
            time_values[currentWindow][2].append(unconverted_time) # In the 100 ms after

        if currentWindow + 1 < len(tap_windows):
            window2 = tap_windows[currentWindow+1]
            # Since the before/after windows of width delta can overlap
            if time >= window2[0] - delta and time < window2[0]:
                while currentWindow + 1 >= len(time_values):
                    time_values.append([ [],[],[] ])
                time_values[currentWindow+1][0].append(unconverted_time)
        
        while currentWindow < len(tap_windows) and time > tap_windows[currentWindow][-1] + delta:
            currentWindow += 1

    return time_values

def get_feature_vector(user_id: str, session: str) -> List[List[float]]:
    """
    This will create a list of motion-related features characterizing the given user,
    given a task they perform according to Activity.csv (such as "Reading+Walking")
    """

    #Find the time windows during which the reader is doing the desired task
    activity_data = read_file(user_id, session, 'Activity.csv', \
                            ['Relative_Start_time', 'Relative_End_time', 'TaskID'])
    
    tap_windows = []
    task_for_each_window = []

    for task_name in task_names:
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
            #return []
            continue

        tap_windows_for_this_task = get_tap_events(user_id, session, windows_to_consider)
        tap_windows += tap_windows_for_this_task
        task_for_each_window += ([task_name] * len(tap_windows_for_this_task))


    data = get_user_session_data(user_id, session)

    # Compute the relevant time values for each file (accelerometer and gyroscope)
    time_values_for_each_file = {}
    data_xy_for_each_file = {}
    for file_name in file_names:
        x = x_columns[file_name]
        time_values_for_each_file[file_name] = time_values_during_taps(user_id, session, data, x, tap_windows)
        data_xy_for_each_file[file_name] = read_file(user_id, session, file_name, \
            [x] + y_columns[file_name] + other_columns[file_name])

    min_num_taps = len(tap_windows)
    for file_name in file_names:
        num_taps = len(time_values_for_each_file[file_name])
        if min_num_taps > num_taps:
            min_num_taps = num_taps

    #A feature vector for each tap, to be filled in subsequently:
    features = [[] for i in range(min_num_taps)]

    for window_index in range(min_num_taps):

        task_name = task_for_each_window[window_index]
        #Just once we'll add in user ID, session, task name, and orientation
        features[window_index] += [user_id, session, task_name]
        #Determine orientation from the closest time stamp in the first file (accel). Bit of
        #a kludge, but necessary, since they don't include orientation in the
        #activity file.
        first_file = file_names[0]
        x_column_first_file = x_columns[first_file]
        x_values_first_file = list(data.loc[:, x_column_first_file])

        if len(x_values_first_file) == 0:
            return None

        start_of_window = tap_windows[window_index][0]
        closest_index = len(x_values_first_file) - 1
        for i in range(len(x_values_first_file)):
            if x_values_first_file[i] > start_of_window:
                closest_index = i
                break
        for c in other_columns[first_file]:
            c_value = data[c][closest_index]
            features[window_index].append(c_value)

        for file_name in file_names:
            x = x_columns[file_name]
            time_values = time_values_for_each_file[file_name]
            data_xy = data_xy_for_each_file[file_name]

            # Here we look at one particular tap, given by window
            window = time_values[window_index]
            before = window[0]
            during = window[1]
            after = window[2]

            if len(before) == 0 or len(after) == 0:
                # This happens sometimes at the very end.
                features[window_index] += [None] * 8 * len(y_columns[file_name])
                break # Just skip data missing its before or after

            if (before[0] != 2214882326000):
                continue

            # New feature for speed: create a slice of the dataframe
            # to focus on when finding values
            x_values = list(data.loc[:, x])
            start_index = x_values.index(before[0])
            #end_index = x_values.index(after[-1]) + 1
            end_index = start_index + len(before) + len(during) + len(after)
            window_data = data_xy[start_index:end_index]

            # Now we find the relevant values at each point in time
            before_values_for_each_y = {}
            during_values_for_each_y = {}
            after_values_for_each_y = {}
            #Initialize everything
            for y in y_columns[file_name]:
                before_values_for_each_y[y] = []
                during_values_for_each_y[y] = []
                after_values_for_each_y[y] = []
            index = start_index
            for k in range(len(before)):
                for y in y_columns[file_name]:
                    before_values_for_each_y[y].append(window_data[y][index])
                index += 1
            for k in range(len(during)):
                for y in y_columns[file_name]:
                    during_values_for_each_y[y].append(window_data[y][index])
                index += 1
            for k in range(len(after)):
                for y in y_columns[file_name]:
                    after_values_for_each_y[y].append(window_data[y][index])
                index += 1

            if len(during) < 2:
                # If there were none or one measurements during the tap,
                # add the closest ones
                during = [before[-1]] + during + [after[0]]
                for y in y_columns[file_name]:
                    during_values_for_each_y[y] = [before_values_for_each_y[y][-1]] + \
                        during_values_for_each_y[y] + [after_values_for_each_y[y][0]]

            for y in y_columns[file_name]:

                before_values = before_values_for_each_y[y]
                during_values = during_values_for_each_y[y]
                after_values = after_values_for_each_y[y]

                # Feature 1
                mean_during = mean(during_values)

                # Feature 2
                sd_during = sd(during_values)

                # Feature 3
                mean_before = mean(before_values)
                mean_after = mean(after_values)
                difference_before_after = mean_after - mean_before

                # Feature 4
                net_change_due_to_tap = mean_during - mean_before

                # Feature 5
                max_tap = max(during_values)
                max_change = max_tap - mean_before

                # Feature 6
                avgDiffs = []
                for j in range(len(after)):
                    subsequentValues = after[j:]
                    subsequentDistances = list(map(lambda x: abs(x - mean_before), subsequentValues))
                    averageDistance = mean(subsequentDistances)
                    avgDiffs.append(averageDistance)
                time_of_earliest_restoration = min(avgDiffs)
                restoration_time = time_of_earliest_restoration - during[-1]

                # Feature 7
                t_before_center = (before[0] + before[-1]) / 2 
                t_after_center = (after[0] + after[-1]) / 2
                normalized_duration = (t_after_center - t_before_center) / (mean_after - mean_before)
                
                # Feature 8
                t_max_in_tap = during[during_values.index(max_tap)]
                normalized_duration_max = (t_after_center - t_max_in_tap) / (mean_after - max_tap)

                features[window_index] += [mean_during, sd_during, difference_before_after,
                                    net_change_due_to_tap, max_change, restoration_time,
                                    normalized_duration, normalized_duration_max]

                # We will make a few plots as we go
                if (before[0] == 2214882326000): #random.choice(range(5000)) == 1):
                    print("Making a plot at time " + str(before[0]))
                    beforePoints, = plt.plot(before, before_values, 'bo', color = 'blue', label = 'Before Tap')
                    afterPoints, = plt.plot(after, after_values, 'bo', color = 'red', label = 'After Tap')
                    duringPoints, = plt.plot(during, during_values, 'bo', color = 'black', label = 'During Tap')
                    plt.xlabel('Event Time')
                    plt.ylabel(y)

                    plt.legend(handles = [beforePoints, duringPoints, afterPoints])

                    min_x = before[0] - (before[1] - before[0]) * 10
                    min_y = min([min(during_values), min(before_values), min(after_values)])
                    # Mark the mean during tap event (Feature 1)
                    plt.hlines(y = mean_during, xmin = min_x, xmax = during[-1], linestyle='dashed', \
                                color='black')
                    plt.annotate(xy = (min_x, mean_during), s = 'avgDuringTap')
                    # Mark the mean before
                    plt.hlines(y = mean_before, xmin = min_x, xmax = before[-1], linestyle='dashed', \
                                color='blue')
                    plt.annotate(xy = (min_x, mean_before), s = 'avg100msBefore')
                    # Mark the mean after
                    plt.hlines(y = mean_after, xmin = min_x, xmax = after[-1], linestyle='dashed', \
                                color='red')
                    plt.annotate(xy = (min_x, mean_after), s = 'avg100msAfter')
                    # Mark the max in the tap
                    plt.hlines(y = max_tap, xmin = min_x, xmax = t_max_in_tap, linestyle='dashed', \
                                color='black')
                    plt.annotate(xy = (min_x, max_tap), s = 'maxTap')

                    # Mark the important points
                    plt.vlines(x = t_before_center, ymin = min_y, ymax = mean_before, linestyle='dashed', \
                                color='blue')
                    plt.annotate(xy = (t_before_center, min_y), s = 'tBeforeCenter')
                    plt.vlines(x = t_after_center, ymin = min_y, ymax = mean_after, linestyle='dashed', \
                                color='red')
                    plt.annotate(xy = (t_after_center, min_y), s = 'tAfterCenter')
                    plt.vlines(x = t_max_in_tap, ymin = min_y, ymax = max_tap, linestyle='dashed', \
                                color='black')
                    plt.annotate(xy = (t_max_in_tap, min_y), s = 'tMax')
                    # plt.vlines(x = time_of_earliest_restoration, ymin = min_y, ymax = mean_before, linestyle='dashed', \
                    #             color = 'red')
                    # plt.annotate(xy = (time_of_earliest_restoration, min_y), s = 'tRestoration')

                    plt.savefig('Plots/Project/'+session+'_'+y+'_time_'+str(before[0]))
                    plt.close()

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
    os.mkdir(folder_name)
    for user in get_user_ids():
        print("Analysis of user: " + user)
        subfolder_name = folder_name + "/" + user
        os.mkdir(subfolder_name)
        for session in get_user_session_ids(user):
            print("Session: " + session)
            file_name = subfolder_name + "/" + session + ".csv"
            data = get_feature_vector(user, session)
            if data == None:
                continue
            if label:
                data = [labels] + data
            write_to_csv(data, file_name)

# Good plot here:
# 865501_session_22_gyroY_time_2214882326000

def write_random_user():
    folder_name = "TapData"
    label = False
    user = '865501' #random.choice(get_user_ids())
    print("Analysis of user: " + user)
    subfolder_name = folder_name + "/" + user
    #os.mkdir(subfolder_name)
    for session in get_user_session_ids(user):
        if session != '865501_session_22':
            continue
        print("Session: " + session)
        file_name = subfolder_name + "/" + session + ".csv"
        data = get_feature_vector(user, session)
        if data == None:
            continue
        if label:
            data = [labels] + data
        write_to_csv(data, file_name)