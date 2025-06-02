from sklearn_preprocessing import labelEncoder
import pandas as pd
import numpy as np

def preprocessing(df):
  df = df.dropna(subset=['Rank_1', 'Rank_2', 'Odd_1', 'Odd_2', 'Surface']) #drops any rows without ranks, odds or a surface

  """
  for my first version im using the following 9 attributes:
  player id's (p1 and p2), surface, tournament, series, court, rank difference, odds ratio
  and the points difference

  in the next version im going to use score as in the data its given as '6-2 6-4' for example
  we can extract a lot of info from the score but getting that info complicates things so its more of a next
  version thing
  """


  # player encoder
  player_encoder = labelEncoder()
  all_players = pd.concat([df['Player_1'], df['Player_2']]) # joins all the players together to be id'd with a number
  player_encoder.fit(all_players) # id's each player
  df['P1_id'] = player_encoder.tranform(df['Player_1']) 
  df['P2_id'] = player_encoder.transform(df['Player_2'])
  # changes the dataframe to use the id's not the names

  #surface, tournament, series, court encoder
  surface_encoder = labelEncoder()
  df['Surface_id'] = surface_encoder.fit_transform(df['Surface'])
  tournament_encoder = labelEncoder()
  df['Tournament_id'] = tournament_encoder.fit_transform(df['Tournament'])
  series_encoder = labelEncoder()
  df['Series_id'] = series_encoder.fit_transform(df['Series'])
  court_encoder = labelEncoder()
  df['Court_id'] = court_encoder.fit_transform(df['Court'])

  # winner
  df['Label'] = (df['Winner'] == df['Player_1']).astype(int)

  # rank difference, odds ratio and points difference
  df['rank_diff'] = df['Rank_2'] - df['Rank_1']
  df['odds_ratio'] = df['Odd_2'] / df['Odd_1']
  df['pts_diff'] = df['Pts_1'] - df['Pts_2']

  # all our attributes which are used in the nn
  features = df[['P1_id', 'P2_id', 'Surface_id', 'Tournament_id', 'Series_id', 'Court_id', 'rank_diff', 'odds_ratio', 'pts_diff']]
  # who wins prediction label
  labels = df['label']

  return features, labels, player_encoder, surface_encoder, tournament_encoder, series_encoder, court_encoder

