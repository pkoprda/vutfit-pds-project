#!/usr/bin/env python3

import argparse
import re
import pandas as pd


def parse_arguments():
  parser = argparse.ArgumentParser(description='Log Monitor')
  parser.add_argument('-training', type=argparse.FileType('r'), metavar='FILE', help='Training file')
  parser.add_argument('-testing', type=argparse.FileType('r'), metavar='FILE', help='Testing file')
  return parser.parse_args()

def create_dataframe(file_path):
  with open(file_path) as f:
    lines = f.readlines()

  # Create dataframe
  df = pd.DataFrame([re.split(r'\s-\s', line.strip(), maxsplit=2) for line in lines], columns=['DateTime', 'LevelNode', 'Content'])

  # Perform data transformation
  df[['Date', 'Time']] = df['DateTime'].str.split(' ', expand=True)
  df[['Level', 'Node']] = df['LevelNode'].str.split(' ', n=1, expand=True)
  df['Node'] = df['Node'].str.replace(r'\[/?','', regex=True).str.replace(r'\]', '', regex=True)
  df[['Node', 'Component']] = df['Node'].str.split('[:]', n=1, expand=True)
  df[['Component', 'Id']] = df['Component'].str.split('@', n=1, expand=True)

  # Remove unnecessary columns
  df = df.drop(columns=['DateTime', 'LevelNode'])

  # Reorder columns
  df = df[['Date', 'Time', 'Level', 'Node', 'Component', 'Id', 'Content']]

  return df


if __name__ == '__main__':
  args = parse_arguments()

  if args.training:
    df_train = create_dataframe(file_path=args.training.name)
  
  if args.testing:
    df_test = create_dataframe(file_path=args.testing.name)
