#!/usr/bin/env python3

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import argparse
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest


# Regular expressions and their replacement strings
# Retrieved from https://github.com/logpai/loghub/blob/master/Zookeeper/Zookeeper_2k.log_templates.csv
patterns = {
  r'.+ GOODBYE \/.*:.* .+': 'E1',
  r'Accepted socket connection from \/.*:.*': 'E2',
  r'autopurge.purgeInterval set to .*': 'E3',
  r'autopurge.snapRetainCount set to .*': 'E4',
  r'Cannot open channel to .* at election address \/.*:.*': 'E5',
  r'caught end of stream exception': 'E6',
  r'Client attempting to establish new session at \/.*:.*': 'E7',
  r'Client attempting to renew session .* at \/.*:.*': 'E8',
  r'Closed socket connection for client \/.*:.* \(no session established for client\)': 'E9',
  r'Closed socket connection for client \/.*:.* which had sessionid .*': 'E10',
  r'Connection broken for id .*, my id = .*, error =': 'E11',
  r'Connection request from old client \/.*:.*; will be dropped if server is in r-o mode': 'E12',
  r'Established session .* with negotiated timeout .* for client \/.*:.*': 'E13',
  r'Exception causing close of session .* due to java.io.IOException: ZooKeeperServer not running': 'E14',
  r'Expiring session .*, timeout of .*ms exceeded': 'E15',
  r'First is .*': 'E16',
  r'Follower sid: .* : info : org\.apache\.zookeeper\.server\.quorum\.QuorumPeer\$QuorumServer\@.*': 'E17',
  r'FOLLOWING - LEADER ELECTION TOOK - .*': 'E19',
  r'Getting a snapshot from leader': 'E20',
  r'Got user-level KeeperException when processing sessionid:.* type:create cxid:.* zxid:.* txntype:.* reqpath:.* Error Path:.* Error:KeeperErrorCode = NodeExists for .*': 'E21',
  r'Have quorum of supporters; starting up and setting last processed zxid: .*': 'E22',
  r'Have smaller server identifier, so dropping the connection: \(.*, .*\)': 'E23',
  r'Interrupted while waiting for message on queue': 'E24',
  r'Interrupting SendWorker': 'E25',
  r'maxSessionTimeout set to .*': 'E27',
  r'minSessionTimeout set to .*': 'E28',
  r'My election bind port: .*/.*:.*': 'E29',
  r'New election. My id =  .*, proposed zxid=.*': 'E30',
  r'Notification time out: .*': 'E31',
  r'Notification: .* \(n\.leader\), .* \(n\.zxid\), .* \(n\.round\), FOLLOWING \(n.state\), .* \(n.sid\), .* \(n.peerEPoch\), FOLLOWING \(my state\)': 'E32',
  r'Notification: .* \(n\.leader\), .* \(n\.zxid\), .* \(n\.round\), FOLLOWING \(n.state\), .* \(n.sid\), .* \(n.peerEPoch\), LEADING \(my state\)': 'E33',
  r'Notification: .* \(n\.leader\), .* \(n\.zxid\), .* \(n\.round\), LEADING \(n.state\), .* \(n.sid\), .* \(n.peerEPoch\), LOOKING \(my state\)': 'E34',
  r'Notification: .* \(n\.leader\), .* \(n\.zxid\), .* \(n\.round\), LOOKING \(n.state\), .* \(n.sid\), .* \(n.peerEPoch\), FOLLOWING \(my state\)': 'E35',
  r'Notification: .* \(n\.leader\), .* \(n\.zxid\), .* \(n\.round\), LOOKING \(n.state\), .* \(n.sid\), .* \(n.peerEPoch\), LEADING \(my state\)': 'E36',
  r'Notification: .* \(n\.leader\), .* \(n\.zxid\), .* \(n\.round\), LOOKING \(n.state\), .* \(n.sid\), .* \(n.peerEPoch\), LOOKING \(my state\)': 'E37',
  r'Processed session termination for sessionid: .*': 'E38',
  r'Reading snapshot .*': 'E39',
  r'Received connection request \/.*:.*': 'E40',
  r'Revalidating client: .*': 'E41',
  r'Send worker leaving thread': 'E42',
  r'Sending DIFF': 'E43',
  r'Server environment:.*': 'E44',
  r'shutdown of request processor complete': 'E45',
  r'Snapshotting: .* to .*': 'E46',
  r'Starting quorum peer': 'E47',
  r'tickTime set to .*': 'E48',
  r'Unexpected exception causing shutdown while sock still open': 'E49',
  r'Unexpected Exception:': 'E50',
  r'FOLLOWING': 'E18',
  r'LOOKING': 'E26',
}


def parse_arguments():
  parser = argparse.ArgumentParser(description='Log Monitor')
  parser.add_argument('-training', type=argparse.FileType('r'), metavar='FILE', required=True, help='Training file')
  parser.add_argument('-testing', type=argparse.FileType('r'), metavar='FILE', help='Testing file')
  parser.add_argument('-contamination', type=float, metavar='VALUE', default=0.25, help='Contamination value for the model training')
  parser.add_argument('-threshold', type=float, metavar='VALUE', default=0.7, help='Threshold value for testing for anomalies')
  return parser.parse_args()

def create_dataframe(file_path):
  with open(file_path) as f:
    lines = f.readlines()

  # Create dataframe
  df = pd.DataFrame([re.split(r'\s-\s', line.strip(), maxsplit=2) for line in lines], columns=['DateTime', 'LevelNode', 'Content'])

  # Perform data transformation
  df[['Level', 'Node']] = df['LevelNode'].str.split(' ', n=1, expand=True)
  df['Node'] = df['Node'].str.lstrip().replace(r'^\[|\]$', '', regex=True)
  df[['Node', 'Component']] = df['Node'].str.split(':', n=1, expand=True)
  df[['Component', 'Id']] = df['Component'].str.split('@', n=1, expand=True)
  df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S,%f')
  df['Level'] = df['Level'].astype('category')

  # Remove unnecessary columns
  df = df.drop(columns=['LevelNode'])

  # Reorder columns
  df = df[['Content', 'Level', 'Node', 'Component', 'Id', 'DateTime']]

  return df

def create_log_events(df):
  # Replace patterns with their corresponding replacement strings
  for pattern, replacement in patterns.items():
    df.loc[:, 'Content'] = df['Content'].str.replace(pattern, replacement, regex=True)

  # Remove the remaining log events that do not match the patterns
  df = df[df['Content'].str.match(r'^E\d+$')].reset_index(drop=True)
  
  return df

# Define the time window
def get_time_window():
  return pd.Timedelta('1 day')

def create_event_count_matrix(df, testing_model=False):
  event_counts = df.groupby(['Content', 'Level'], observed=False).apply(lambda x: x.set_index('DateTime').resample(get_time_window()).size(), include_groups=False)

  # Convert the Series to a DataFrame
  event_counts_df = event_counts.to_frame(name='Count')

  # Reset the index to make the 'DateTime' column accessible
  event_counts_df.reset_index(inplace=True)

  # Pivot the DataFrame
  X = event_counts_df.pivot_table(index='DateTime', columns='Content', values='Count', fill_value=0, aggfunc=np.sum)

  # Fill missing values with 0 and reset the index
  X.fillna(0, inplace=True)
  X.reset_index(inplace=True)

  # Define the desired column order
  column_order = ['DateTime'] + ['E{}'.format(i) for i in range(1, 51)]

  # Insert 0 values, if some columns does not exist in testing model
  if testing_model:
    for column in column_order:
      if column not in X.columns:
        X[column] = 0
    

  return X[column_order]

def detect_anomalies(model, data, threshold):
    # Predict anomaly scores for the data
    anomaly_scores = model.decision_function(data)
    
    # Extract anomalies based on the threshold
    anomalies = data[anomaly_scores < threshold]
    
    # Return the anomaly scores for each anomaly
    anomaly_scores = anomaly_scores[anomaly_scores < threshold]
    
    return anomalies, anomaly_scores


if __name__ == '__main__':
  args = parse_arguments()
  if args.contamination <= 0 or args.contamination > 0.5:
    raise ValueError("Contamination value should be in interval (0, 0.5]")

  if args.training:
    # Create the dataframe from the file with training data
    df_train = create_dataframe(file_path=args.training.name)

    # Create the log events and create the event count matrix
    df_train_events = create_log_events(df_train)
    X_train = create_event_count_matrix(df_train_events).drop(columns=['DateTime'])

    # Normalize the data using Min-Max scaling
    scaler = MinMaxScaler()
    X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])

    # Train the Isolation Forest model
    model = IsolationForest(contamination=args.contamination)
    model.fit(X_train)
  
  if args.testing:
    # Create the dataframe from the file with testing data
    df_test = create_dataframe(file_path=args.testing.name)

    # Create the event count matrix
    df_test_events = create_log_events(df_test)
    X_test = create_event_count_matrix(df_test_events, testing_model=True)
    
    # DateTime list will be used for the output
    X_datetimes = X_test['DateTime'].tolist()
    X_test = X_test.drop(columns=['DateTime'])

    # Normalize the data using Min-Max scaling
    X_test[X_test.columns] = scaler.fit_transform(X_test[X_test.columns])

    # Detect anomalies
    anomalies, anomaly_scores = detect_anomalies(model, X_test, args.threshold)

    # Print or process the anomalies and their scores as needed
    for datetime, score in zip(X_datetimes, anomaly_scores):
        print(f"Anomaly score of {datetime}: {score}")
