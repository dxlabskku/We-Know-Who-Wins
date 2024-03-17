import re
import pandas as pd 
import json
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
import torch


class load_and_preprocess:
    def __init__(self, directory, minute = None):
        self.directory = directory
        self.min = minute

    def extract_json_from_html(self):
        html_file = open(self.directory, 'r')
        html = html_file.read()
        html_file.close()
        regex_pattern = r'(?<=require\.config\.params\["args"\].=.)[\s\S]*?;'
        data_txt = re.findall(regex_pattern, html)[0]

        # add quotations for json parser
        data_txt = data_txt.replace('matchId', '"matchId"')
        data_txt = data_txt.replace('matchCentreData', '"matchCentreData"')
        data_txt = data_txt.replace('matchCentreEventTypeJson', '"matchCentreEventTypeJson"')
        data_txt = data_txt.replace('formationIdNameMappings', '"formationIdNameMappings"')
        data_txt = data_txt.replace('};', '}')
        data = json.loads(data_txt)

        return data
    
    def find_closest_smaller_key(self, dictionary, input_value):
        closest_key = 0
        closest_value = float('-inf')

        for key, value in dictionary.items():
            if int(key) < input_value and int(key) > closest_key:
                closest_key = int(key)
                closest_value = value

        if closest_key is not None:
            return int(closest_key), float(closest_value)
        else:
            return None

    def sum_values_smaller_or_equal(self, dictionary, input_key):
        total_sum = 0

        for key, value in dictionary.items():
            if int(key) <= input_key:
                total_sum += value

        return total_sum

    def extract_data_from_dict(self):
        # load data from json
        data = self.extract_json_from_html()
        events_dict = data["matchCentreData"]["events"]
        teams_dict = {data["matchCentreData"]['home']['teamId']: data["matchCentreData"]['home']['name'],
                    data["matchCentreData"]['away']['teamId']: data["matchCentreData"]['away']['name']}
        # create non-pass dataframe
        df = pd.DataFrame(events_dict)
        df = df[df['minute'] <= self.min]
        df['eventType'] = df.apply(lambda row: row['type']['displayName'], axis=1)
        df['outcomeType'] = df.apply(lambda row: row['outcomeType']['displayName'], axis=1)
        nonpasses_ids = df.index[~df['eventType'].isin(['Pass', 'Start', 'End', 'FormationSet'])]
        df_non_passes = df.loc[nonpasses_ids, ["id", "minute", "second", "x", "y", "endX", "endY", "teamId", "playerId", "eventType", "outcomeType"]]
        df_non_passes = df_non_passes[df_non_passes['outcomeType'] == 'Successful'].reset_index(drop = True)
        df_pivot = df_non_passes.pivot_table(index = 'playerId', columns = 'eventType', aggfunc='size', fill_value=0).reset_index()
        df_pivot.columns.name = None
        df_pivot['playerId'] = df_pivot['playerId'].astype(int)

        # create players dataframe
        players_home_df = pd.DataFrame(data["matchCentreData"]['home']['players'])
        players_home_df["teamId"] = data["matchCentreData"]['home']['teamId']
        players_home_df['height'][players_home_df.height == 0] = np.nan
        players_home_df['height'] = players_home_df['height'].fillna(players_home_df['height'].mean())
        players_home_df['weight'][players_home_df.weight == 0] = np.nan
        players_home_df['weight'] = players_home_df['weight'].fillna(players_home_df['weight'].mean())
        players_away_df = pd.DataFrame(data["matchCentreData"]['away']['players'])
        players_away_df["teamId"] = data["matchCentreData"]['away']['teamId']
        players_away_df['height'][players_away_df.height == 0] = np.nan
        players_away_df['height'] = players_away_df['height'].fillna(players_away_df['height'].mean())
        players_away_df['weight'][players_away_df.weight == 0] = np.nan
        players_away_df['weight'] = players_away_df['weight'].fillna(players_away_df['weight'].mean())
        players_df = pd.concat([players_home_df, players_away_df])
        players_df = players_df[players_df['isFirstEleven'] == True].reset_index(drop = True)
        players_df = players_df[['playerId', 'shirtNo', 'name', 'position', 'height', 'weight', 'stats', 'subbedOutPeriod', 'subbedOutExpandedMinute', 'teamId', 'isFirstEleven']]

        stats = players_df['stats'].tolist()
        common_keys = set(stats[0].keys())
        for dictionary in stats[1:]:
            common_keys.intersection_update(dictionary.keys())
        players_df['stats'] = players_df['stats'].apply(lambda d: {key: value for key, value in d.items() if key in common_keys})
        ratings = []
        touches = []
        passesTotal = []
        passesAccurate = []
        subs = []
        
        for i in range(len(players_df)):
            lmin, rating = self.find_closest_smaller_key(players_df['stats'][i]['ratings'], self.min)
            ratings.append(rating)
            if 'touches' not in players_df['stats'][i].keys():
                touches.append(0)
            else:
                touch = self.sum_values_smaller_or_equal(players_df['stats'][i]['touches'], lmin)
                touches.append(touch)
            if 'passesTotal' not in players_df['stats'][i].keys() or 'passesAccurate' not in players_df['stats'][i].keys():
                passesTotal.append(0)
                passesAccurate.append(0)
            else:
                pt = self.sum_values_smaller_or_equal(players_df['stats'][i]['passesTotal'], lmin)
                passesTotal.append(pt)
                pa = self.sum_values_smaller_or_equal(players_df['stats'][i]['passesAccurate'], lmin)
                passesAccurate.append(pa)
            if players_df['subbedOutExpandedMinute'][i] <= lmin:
                subs.append(players_df['subbedOutExpandedMinute'][i])
            else:
                subs.append(0)
                
        players_df['ratings'] = ratings 
        players_df['touches'] = touches 
        players_df['passesTotal'] = passesTotal
        players_df['passesAccurate'] = passesAccurate
        players_df['subs'] = subs

        players_df = players_df.drop(['subbedOutPeriod', 'subbedOutExpandedMinute'], axis = 1)
        players_df['paRate'] = players_df['passesAccurate'] / players_df['passesTotal']
        players_df['paRate'] = players_df['paRate'].fillna(0)

        players_df = players_df.merge(df_pivot, on = 'playerId')

        players_home_df = players_df[players_df['teamId'] == data["matchCentreData"]['home']['teamId']]
        players_away_df = players_df[players_df['teamId'] == data["matchCentreData"]['away']['teamId']]

        players_ids = data["matchCentreData"]["playerIdNameDictionary"]

        return events_dict, players_df, teams_dict, players_home_df, players_away_df

    def get_passes_df(self):
        events_dict, players_df, teams_dict, _, _ = self.extract_data_from_dict()
        
        df = pd.DataFrame(events_dict)
        df['eventType'] = df.apply(lambda row: row['type']['displayName'], axis=1)
        df['outcomeType'] = df.apply(lambda row: row['outcomeType']['displayName'], axis=1)

        # create receiver column based on the next event
        # this will be correct only for successfull passes
        df["receiver"] = df["playerId"].shift(-1)
        df = df[df['outcomeType'] == 'Successful'].reset_index(drop = True)

        # filter only passes
        passes_ids = df.index[df['eventType'] == 'Pass']

        df_passes = df.loc[
            passes_ids, ["id", "minute", "second", "x", "y", "endX", "endY", "teamId", "playerId", "receiver", "eventType", "outcomeType"]]

        return df_passes
    
    def get_passes_between_df(self):

        data = self.extract_json_from_html()
        matchId = data['matchId']
        events_dict, players_df, teams_dict, home_players_df, away_players_df = self.extract_data_from_dict()
        passes_df = self.get_passes_df()
        home_id = data["matchCentreData"]['home']['teamId']
        away_id = data["matchCentreData"]['away']['teamId']
        passes_df = passes_df[passes_df['minute'] <= self.min]

        # filter for only team
        home_passes_df = passes_df[passes_df["teamId"] == home_id]

        # add column with first eleven players only
        home_passes_df = home_passes_df.merge(players_df[["playerId", "isFirstEleven"]], on='playerId', how='left')
        # filter on first eleven column
        home_passes_df = home_passes_df[home_passes_df['isFirstEleven'] == True]

        # calculate mean positions for players
        home_average_locs_and_count_df = (home_passes_df.groupby('playerId')
                                    .agg({'x': ['mean'], 'y': ['mean', 'count']}))
        home_average_locs_and_count_df.columns = ['x', 'y', 'count']
        home_average_locs_and_count_df = home_average_locs_and_count_df.merge(players_df[['playerId', 'name', 'shirtNo', 'position']],
                                                                on='playerId', how='left')
        home_average_locs_and_count_df = home_average_locs_and_count_df.set_index('playerId')

        # calculate the number of passes between each position (using min/ max so we get passes both ways)
        home_passes_player_ids_df = home_passes_df.loc[:, ['id', 'playerId', 'receiver', 'teamId']]

        # get passes between each player
        home_passes_between_df = home_passes_player_ids_df.groupby(['playerId', 'receiver']).id.count().reset_index()
        home_passes_between_df.rename({'id': 'pass_count'}, axis='columns', inplace=True)

        # add on the location of each player so we have the start and end positions of the lines
        home_passes_between_df = home_passes_between_df.merge(home_average_locs_and_count_df, left_on='playerId', right_index=True)
        home_passes_between_df = home_passes_between_df.merge(home_average_locs_and_count_df, left_on='receiver', right_index=True,
                                                suffixes=['', '_end'])

        away_passes_df = passes_df[passes_df["teamId"] == away_id]

        # add column with first eleven players only
        away_passes_df = away_passes_df.merge(players_df[["playerId", "isFirstEleven"]], on='playerId', how='left')
        # filter on first eleven column
        away_passes_df = away_passes_df[away_passes_df['isFirstEleven'] == True]

        # calculate mean positions for players
        away_average_locs_and_count_df = (away_passes_df.groupby('playerId')
                                    .agg({'x': ['mean'], 'y': ['mean', 'count']}))
        away_average_locs_and_count_df.columns = ['x', 'y', 'count']
        away_average_locs_and_count_df = away_average_locs_and_count_df.merge(players_df[['playerId', 'name', 'shirtNo', 'position']],
                                                                on='playerId', how='left')
        away_average_locs_and_count_df = away_average_locs_and_count_df.set_index('playerId')

        # calculate the number of passes between each position (using min/ max so we get passes both ways)
        away_passes_player_ids_df = away_passes_df.loc[:, ['id', 'playerId', 'receiver', 'teamId']]

        # get passes between each player
        away_passes_between_df = away_passes_player_ids_df.groupby(['playerId', 'receiver']).id.count().reset_index()
        away_passes_between_df.rename({'id': 'pass_count'}, axis='columns', inplace=True)

        # add on the location of each player so we have the start and end positions of the lines
        away_passes_between_df = away_passes_between_df.merge(away_average_locs_and_count_df, left_on='playerId', right_index=True)
        away_passes_between_df = away_passes_between_df.merge(away_average_locs_and_count_df, left_on='receiver', right_index=True,
                                                suffixes=['', '_end'])
        
        for i in home_passes_between_df.index.tolist():
            if home_passes_between_df['name'][i] == home_passes_between_df['name_end'][i]:
                home_passes_between_df = home_passes_between_df.drop(i)

        for i in away_passes_between_df.index.tolist():
            if away_passes_between_df['name'][i] == away_passes_between_df['name_end'][i]:
                away_passes_between_df = away_passes_between_df.drop(i)
        
        result = []
        result_1 = data['matchCentreData']['htScore']
        if int(result_1[0]) > int(result_1[4]):
            result_1 = 0
        elif int(result_1[0]) < int(result_1[4]):
            result_1 = 1
        elif int(result_1[0]) == int(result_1[4]):
            result_1 = 2
        
        result.append(result_1)

        result_2 = data['matchCentreData']['ftScore']
        if int(result_2[0]) > int(result_2[4]):
            result_2 = 0
        elif int(result_2[0]) < int(result_2[4]):
            result_2 = 1
        elif int(result_2[0]) == int(result_2[4]):
            result_2 = 2
        
        result.append(result_2)

        date = data['matchCentreData']['timeStamp'][:10]

        return home_passes_between_df, away_passes_between_df, home_average_locs_and_count_df, away_average_locs_and_count_df, home_players_df, away_players_df, result, matchId, date
    

class preprocess_home_data:
    def __init__(self, directory, files, minute):
        self.directory = directory
        self.onlyfiles = files
        self.min = minute

    def preprocess(self):
        results_home = []
        match_home = []
        xs_home = []
        xs2_home = []
        edge_indices_home = []
        edge_attributes_home = []
        flag_home = []
        cnt = 0
        for htmls in tqdm(self.onlyfiles):
                cnt += 1
                lp =  load_and_preprocess(f'{self.directory}{htmls}', self.min)
                try:
                    a, b, c, d, e, f, g, h, date = lp.get_passes_between_df()
                    match_home.append(h)
                    condition = lambda x : 1 if x > 0 else 0
                    e['subs'] = e['subs'].apply(condition)
                    x_temp = e.merge(c[['name', 'x', 'y']], on = 'name').sort_values(by = 'shirtNo').drop(['name','playerId',  'teamId', 'isFirstEleven'], axis = 1).reset_index(drop = True)
                    x_temp.reset_index(inplace = True)
                    if 'Card' not in x_temp.columns.tolist():
                        x_temp['Card'] = 0
                    if 'OffsidePass' not in x_temp.columns.tolist():
                        x_temp['OffsidePass'] = 0
                    if 'OffsideProvoked' not in x_temp.columns.tolist():
                        x_temp['OffsideProvoked'] = 0
                    if 'Punch' not in x_temp.columns.tolist():
                        x_temp['Punch'] = 0
                    if 'Save' not in x_temp.columns.tolist():
                        x_temp['Save'] = 0
                    if 'MissedShots' not in x_temp.columns.tolist():
                        x_temp['MissedShots'] = 0
                    if 'Tackle' not in x_temp.columns.tolist():
                        x_temp['Tackle'] = 0
                    if 'TakeOn' not in x_temp.columns.tolist():
                        x_temp['TakeOn'] = 0
                    if 'BlockedPass' not in x_temp.columns.tolist():
                        x_temp['BlockedPass'] = 0
                    if 'SavedShot' not in x_temp.columns.tolist():
                        x_temp['SavedShot'] = 0
                    if 'CornerAwarded' not in x_temp.columns.tolist():
                        x_temp['CornerAwarded'] = 0
                    if 'Dispossessed' not in x_temp.columns.tolist():
                        x_temp['Dispossessed'] = 0
                    if 'Interception' not in x_temp.columns.tolist():
                        x_temp['Interception'] = 0
                    
                    x_temp = x_temp[['index', 'shirtNo', 'position', 'height', 'weight', 'ratings', 'touches', 'passesTotal',
                                    'passesAccurate','subs', 'paRate', 'Aerial', 'BallRecovery', 'BallTouch', 'BlockedPass', 
                                    'Card',  'Clearance', 'CornerAwarded', 'Dispossessed', 'Foul', 'Interception', 
                                    'MissedShots', 'Save', 'SavedShot', 'Tackle', 'TakeOn', 'x', 'y', 'OffsidePass', 'OffsideProvoked', 'Punch']]

                    corr = dict(zip(x_temp.shirtNo, x_temp.index))
                    x_i = x_temp.drop(['index', 'shirtNo', 'touches', 'passesTotal', 'passesAccurate','subs',
                                    'Aerial', 'BallRecovery', 'BallTouch', 'BlockedPass', 
                                    'Card',  'Clearance', 'CornerAwarded', 'Dispossessed', 'Foul', 'Interception', 
                                    'MissedShots', 'Save', 'SavedShot', 'Tackle', 'TakeOn', 'OffsidePass', 'OffsideProvoked', 'Punch'], axis = 1).values
                    
                    x_i2 = x_temp[['touches', 'passesTotal', 'passesAccurate','subs', 'Aerial', 'BallRecovery', 'BallTouch', 'BlockedPass', 
                                    'Card',  'Clearance', 'CornerAwarded', 'Dispossessed', 'Foul', 'Interception', 
                                    'MissedShots', 'Save', 'SavedShot', 'Tackle', 'TakeOn','OffsidePass', 'OffsideProvoked', 'Punch']].values
                    
                    if len(x_i) == 11:
                        flag_home.append(False)
                    else:
                        flag_home.append(True)
                    
                    xs_home.append(x_i)
                    xs2_home.append(x_i2)

                    results_home.append(np.array([g]))

                    edge = a.sort_values(by = 'shirtNo')
                    edge['pass_traj'] = edge['position'] + edge['position_end']
                    edge_a = list(edge['shirtNo'])

                    for a in range(len(edge_a)):
                        edge_a[a] = corr[edge_a[a]]

                    edge_b = list(edge['shirtNo_end'])

                    for a in range(len(edge_b)):
                        edge_b[a] = corr[edge_b[a]]

                    edge_index = np.array([edge_a, edge_b])
                    edge_attr = edge[['pass_count']].values

                    if 'Card' not in x_temp.columns.tolist():
                        x_temp['Card'] = 0
                    if 'OffsidePass' not in x_temp.columns.tolist():
                        x_temp['OffsidePass'] = 0
                    if 'OffsideProvoked' not in x_temp.columns.tolist():
                        x_temp['OffsideProvoked'] = 0
                    if 'Punch' not in x_temp.columns.tolist():
                        x_temp['Punch'] = 0
                    if 'Save' not in x_temp.columns.tolist():
                        x_temp['Save'] = 0
                    if 'MissedShots' not in x_temp.columns.tolist():
                        x_temp['MissedShots'] = 0
                    if 'Tackle' not in x_temp.columns.tolist():
                        x_temp['Tackle'] = 0
                    if 'TakeOn' not in x_temp.columns.tolist():
                        x_temp['TakeOn'] = 0
                    if 'BlockedPass' not in x_temp.columns.tolist():
                        x_temp['BlockedPass'] = 0
                    if 'SavedShot' not in x_temp.columns.tolist():
                        x_temp['SavedShot'] = 0
                    if 'CornerAwarded' not in x_temp.columns.tolist():
                        x_temp['CornerAwarded'] = 0
                    if 'Dispossessed' not in x_temp.columns.tolist():
                        x_temp['Dispossessed'] = 0
                    if 'Interception' not in x_temp.columns.tolist():
                        x_temp['Interception'] = 0

                    x_temp = x_temp[['index', 'shirtNo', 'position', 'height', 'weight', 'ratings', 'touches', 'passesTotal',
                                        'passesAccurate','subs', 'paRate', 'Aerial', 'BallRecovery', 'BallTouch', 'BlockedPass', 
                                        'Card',  'Clearance', 'CornerAwarded', 'Dispossessed', 'Foul', 'Interception', 
                                        'MissedShots', 'Save', 'SavedShot', 'Tackle', 'TakeOn', 'x', 'y', 'OffsidePass', 'OffsideProvoked', 'Punch']]
                    x = x_i if cnt == 1 else np.vstack([x, x_temp.drop(['index', 'shirtNo', 'touches', 'passesTotal', 'passesAccurate','subs',
                                                                'Aerial', 'BallRecovery', 'BallTouch', 'BlockedPass', 
                                                                'Card',  'Clearance', 'CornerAwarded', 'Dispossessed', 'Foul', 'Interception', 
                                                                'MissedShots', 'Save', 'SavedShot', 'Tackle', 'TakeOn', 'OffsidePass', 'OffsideProvoked', 'Punch'], axis = 1).values])

                    x2_home = edge_attr if cnt == 1 else np.vstack([x2_home, edge[['pass_count']].values])
            

                    x3 = x_i2 if cnt == 1 else np.vstack([x3, x_temp[['touches', 'passesTotal', 'passesAccurate','subs', 'Aerial', 'BallRecovery', 'BallTouch', 'BlockedPass', 
                                                                        'Card',  'Clearance', 'CornerAwarded', 'Dispossessed', 'Foul', 'Interception', 
                                                                        'MissedShots', 'Save', 'SavedShot', 'Tackle', 'TakeOn','OffsidePass', 'OffsideProvoked', 'Punch']].values])
                    edge_indices_home.append(edge_index)
                    edge_attributes_home.append(edge_attr)
                except:
                    pass

        x3s = []
        for i in range(len(xs_home)):
            x3s.append(list(np.sum(xs2_home[i], axis = 0)))

        df_x3_home = pd.DataFrame(x3s)
        df_x3_home.columns = ['touch', 'total_pass', 'success_pass', 'sub', 
                        'Aerial', 'BallRecovery', 'BallTouch', 'BlockedPass', 'Card', 'Clearance', 'CornerAwarded', 'Dispossessed', 
                        'Foul', 'Interception', 'MissedShots', 'Save', 'SavedShot', 'Tackle', 'TakeOn',
                        'OffsidePass', 'OffsideProvoked', 'Punch']

        df_x3_home = df_x3_home[['touch', 'sub', 
                        'Aerial', 'BallRecovery', 'BlockedPass', 'Card', 'Clearance', 'CornerAwarded', 'Dispossessed', 
                        'Foul', 'Interception', 'MissedShots', 'SavedShot', 'Tackle', 'TakeOn','OffsideProvoked']]
        
        return df_x3_home, results_home, match_home, xs_home, xs2_home, edge_indices_home, edge_attributes_home, flag_home, x2_home
    

class preprocess_away_data:
    def __init__(self, directory, files, minute):
        self.directory = directory
        self.onlyfiles = files
        self.min = minute

    def preprocess(self):
        results_away = []
        match_away = []
        xs_away = []
        xs2_away = []
        edge_indices_away = []
        edge_attributes_away = []
        flag_away = []
        cnt = 0
        for htmls in tqdm(self.onlyfiles):
                cnt += 1
                lp =  load_and_preprocess(f'{self.directory}{htmls}',self.min)
                try:
                    a, b, c, d, e, f, g, h, date = lp.get_passes_between_df()
                    match_away.append(h)
                    condition = lambda x : 1 if x > 0 else 0
                    f['subs'] = f['subs'].apply(condition)
                    x_temp = f.merge(d[['name', 'x', 'y']], on = 'name').sort_values(by = 'shirtNo').drop(['name','playerId',  'teamId', 'isFirstEleven'], axis = 1).reset_index(drop = True)
                    x_temp.reset_index(inplace = True)
                    if 'Card' not in x_temp.columns.tolist():
                        x_temp['Card'] = 0
                    if 'OffsidePass' not in x_temp.columns.tolist():
                        x_temp['OffsidePass'] = 0
                    if 'OffsideProvoked' not in x_temp.columns.tolist():
                        x_temp['OffsideProvoked'] = 0
                    if 'Punch' not in x_temp.columns.tolist():
                        x_temp['Punch'] = 0
                    if 'Save' not in x_temp.columns.tolist():
                        x_temp['Save'] = 0
                    if 'MissedShots' not in x_temp.columns.tolist():
                        x_temp['MissedShots'] = 0
                    if 'Tackle' not in x_temp.columns.tolist():
                        x_temp['Tackle'] = 0
                    if 'TakeOn' not in x_temp.columns.tolist():
                        x_temp['TakeOn'] = 0
                    if 'BlockedPass' not in x_temp.columns.tolist():
                        x_temp['BlockedPass'] = 0
                    if 'SavedShot' not in x_temp.columns.tolist():
                        x_temp['SavedShot'] = 0
                    if 'CornerAwarded' not in x_temp.columns.tolist():
                        x_temp['CornerAwarded'] = 0
                    if 'Dispossessed' not in x_temp.columns.tolist():
                        x_temp['Dispossessed'] = 0
                    if 'Interception' not in x_temp.columns.tolist():
                        x_temp['Interception'] = 0

                    x_temp = x_temp[['index', 'shirtNo', 'position', 'height', 'weight', 'ratings', 'touches', 'passesTotal',
                                    'passesAccurate','subs', 'paRate', 'Aerial', 'BallRecovery', 'BallTouch', 'BlockedPass', 
                                    'Card',  'Clearance', 'CornerAwarded', 'Dispossessed', 'Foul', 'Interception', 
                                    'MissedShots', 'Save', 'SavedShot', 'Tackle', 'TakeOn', 'x', 'y', 'OffsidePass', 'OffsideProvoked', 'Punch']]

                    corr = dict(zip(x_temp.shirtNo, x_temp.index))
                    x_i = x_temp.drop(['index', 'shirtNo', 'touches', 'passesTotal', 'passesAccurate','subs',
                                    'Aerial', 'BallRecovery', 'BallTouch', 'BlockedPass', 
                                    'Card',  'Clearance', 'CornerAwarded', 'Dispossessed', 'Foul', 'Interception', 
                                    'MissedShots', 'Save', 'SavedShot', 'Tackle', 'TakeOn', 'OffsidePass', 'OffsideProvoked', 'Punch'], axis = 1).values
                    
                    x_i2 = x_temp[['touches', 'passesTotal', 'passesAccurate','subs', 'Aerial', 'BallRecovery', 'BallTouch', 'BlockedPass', 
                                    'Card',  'Clearance', 'CornerAwarded', 'Dispossessed', 'Foul', 'Interception', 
                                    'MissedShots', 'Save', 'SavedShot', 'Tackle', 'TakeOn','OffsidePass', 'OffsideProvoked', 'Punch']].values
                    
                    if len(x_i) == 11:
                        flag_away.append(False)
                    else:
                        flag_away.append(True)
                        
                    xs_away.append(x_i)
                    xs2_away.append(x_i2)

                    results_away.append(np.array([g]))

                    edge = b.sort_values(by = 'shirtNo')
                    edge['pass_traj'] = edge['position'] + edge['position_end']
                    edge_a = list(edge['shirtNo'])

                    for a in range(len(edge_a)):
                        edge_a[a] = corr[edge_a[a]]

                    edge_b = list(edge['shirtNo_end'])

                    for a in range(len(edge_b)):
                        edge_b[a] = corr[edge_b[a]]

                    edge_index = np.array([edge_a, edge_b])
                    edge_attr = edge[['pass_count']].values

                    if 'Card' not in x_temp.columns.tolist():
                        x_temp['Card'] = 0
                    if 'OffsidePass' not in x_temp.columns.tolist():
                        x_temp['OffsidePass'] = 0
                    if 'OffsideProvoked' not in x_temp.columns.tolist():
                        x_temp['OffsideProvoked'] = 0
                    if 'Punch' not in x_temp.columns.tolist():
                        x_temp['Punch'] = 0
                    if 'Save' not in x_temp.columns.tolist():
                        x_temp['Save'] = 0
                    if 'MissedShots' not in x_temp.columns.tolist():
                        x_temp['MissedShots'] = 0
                    if 'Tackle' not in x_temp.columns.tolist():
                        x_temp['Tackle'] = 0
                    if 'TakeOn' not in x_temp.columns.tolist():
                        x_temp['TakeOn'] = 0
                    if 'BlockedPass' not in x_temp.columns.tolist():
                        x_temp['BlockedPass'] = 0
                    if 'SavedShot' not in x_temp.columns.tolist():
                        x_temp['SavedShot'] = 0
                    if 'CornerAwarded' not in x_temp.columns.tolist():
                        x_temp['CornerAwarded'] = 0
                    if 'Dispossessed' not in x_temp.columns.tolist():
                        x_temp['Dispossessed'] = 0
                    if 'Interception' not in x_temp.columns.tolist():
                        x_temp['Interception'] = 0

                    x_temp = x_temp[['index', 'shirtNo', 'position', 'height', 'weight', 'ratings', 'touches', 'passesTotal',
                                    'passesAccurate','subs', 'paRate', 'Aerial', 'BallRecovery', 'BallTouch', 'BlockedPass', 
                                    'Card',  'Clearance', 'CornerAwarded', 'Dispossessed', 'Foul', 'Interception', 
                                    'MissedShots', 'Save', 'SavedShot', 'Tackle', 'TakeOn', 'x', 'y', 'OffsidePass', 'OffsideProvoked', 'Punch']]
                    
                    x = x_i if cnt == 1 else np.vstack([x, x_temp.drop(['index', 'shirtNo', 'touches', 'passesTotal', 'passesAccurate','subs',
                                                                'Aerial', 'BallRecovery', 'BallTouch', 'BlockedPass', 
                                                                'Card',  'Clearance', 'CornerAwarded', 'Dispossessed', 'Foul', 'Interception', 
                                                                'MissedShots', 'Save', 'SavedShot', 'Tackle', 'TakeOn', 'OffsidePass', 'OffsideProvoked', 'Punch'], axis = 1).values])

                    x2_away = edge_attr if cnt == 1 else np.vstack([x2_away, edge[['pass_count']].values])

                    x3 = x_i2 if cnt == 1 else np.vstack([x3, x_temp[['touches', 'passesTotal', 'passesAccurate','subs', 'Aerial', 'BallRecovery', 'BallTouch', 'BlockedPass', 
                                                                    'Card',  'Clearance', 'CornerAwarded', 'Dispossessed', 'Foul', 'Interception', 
                                                                    'MissedShots', 'Save', 'SavedShot', 'Tackle', 'TakeOn','OffsidePass', 'OffsideProvoked', 'Punch']].values])
                    edge_indices_away.append(edge_index)
                    edge_attributes_away.append(edge_attr)
                except:
                    pass

        x3s = []
        for i in range(len(xs_away)):
            x3s.append(list(np.sum(xs2_away[i], axis = 0)))

        df_x3_away = pd.DataFrame(x3s)
        df_x3_away.columns = ['touch', 'total_pass', 'success_pass', 'sub', 
                        'Aerial', 'BallRecovery', 'BallTouch', 'BlockedPass', 'Card', 'Clearance', 'CornerAwarded', 'Dispossessed', 
                        'Foul', 'Interception', 'MissedShots', 'Save', 'SavedShot', 'Tackle', 'TakeOn',
                        'OffsidePass', 'OffsideProvoked', 'Punch']

        df_x3_away = df_x3_away[['touch', 'sub', 
                        'Aerial', 'BallRecovery', 'BlockedPass', 'Card', 'Clearance', 'CornerAwarded', 'Dispossessed', 
                        'Foul', 'Interception', 'MissedShots', 'SavedShot', 'Tackle', 'TakeOn','OffsideProvoked']]
        
        return df_x3_away, results_away, match_away, xs_away, xs2_away, edge_indices_away, edge_attributes_away, flag_away, x2_away
    

class get_scaler:
    def __init__(self, flag_home, flag_away, xs_home, xs_away, edge_attributes_home, edge_attributes_away, df_x3_home, df_x3_away, x2_home, x2_away, METHOD = 'min-max'):
        self.flaghome = flag_home 
        self.flagaway = flag_away
        self.delete_home = [i for i, e in enumerate(self.flaghome) if e == True]
        self.delete_away = [i for i, e in enumerate(self.flagaway) if e == True]
        self.xs_home = xs_home 
        self.xs_away = xs_away
        self.edge_attributes_home = edge_attributes_home 
        self.edge_attributes_away = edge_attributes_away 
        self.df_x3_home = df_x3_home
        self.df_x3_away = df_x3_away
        self.x2_home = x2_home
        self.x2_away = x2_away
        self.METHOD = METHOD

    def make_scaler(self):
        delete = list(set(self.delete_home + self.delete_away))
        for i in range(len(self.xs_home)):
            if i not in delete:
                if i == 0:
                    df_x_home = pd.DataFrame(self.xs_home[i])
                    df_x_home.columns = ['position', 'height', 'weight', 'ratings', 'pass_accuracy', 'x', 'y']
                else:
                    df_x_home_i = pd.DataFrame(self.xs_home[i])
                    df_x_home_i.columns = ['position', 'height', 'weight', 'ratings', 'pass_accuracy', 'x', 'y']
                    df_x_home = pd.concat([df_x_home, df_x_home_i]).reset_index(drop = True)
            else:
                pass

        for i in range(len(self.xs_away)):
            if i not in delete:
                if i == 0:
                    df_x_away = pd.DataFrame(self.xs_away[i])
                    df_x_away.columns = ['position', 'height', 'weight', 'ratings', 'pass_accuracy', 'x', 'y']
                else:
                    df_x_away_i = pd.DataFrame(self.xs_away[i])
                    df_x_away_i.columns = ['position', 'height', 'weight', 'ratings', 'pass_accuracy', 'x', 'y']
                    df_x_away = pd.concat([df_x_away, df_x_away_i]).reset_index(drop = True)
            else:
                pass

        df_x = pd.concat([df_x_home, df_x_away], axis = 0).reset_index(drop = True)

        label_encoder = LabelEncoder()
        df_x['position'] = label_encoder.fit_transform(df_x['position'])

        df_x2_home = pd.DataFrame(self.x2_home)
        df_x2_away = pd.DataFrame(self.x2_away)
        df_x2_home.columns = ['pass_cnts']
        df_x2_away.columns = ['pass_cnts']
        df_x2_home = df_x2_home.drop(index = delete).reset_index(drop = True)
        df_x2_away = df_x2_away.drop(index = delete).reset_index(drop = True)
        df_x2 = pd.concat([df_x2_home, df_x2_away], axis = 0).reset_index(drop = True)

        df_x3_home = self.df_x3_home.drop(index = delete)
        df_x3_away = self.df_x3_away.drop(index = delete)
        df_x3 = pd.concat([df_x3_home, df_x3_away], axis = 0).reset_index(drop = True)

        cols_to_normalize1 = ['touch', 'sub', 
                'Aerial', 'BallRecovery', 'BlockedPass', 'Card', 'Clearance', 'CornerAwarded', 'Dispossessed', 
                'Foul', 'Interception', 'MissedShots', 'SavedShot', 'Tackle', 'TakeOn','OffsideProvoked']
        cols_to_normalize2 = ['height', 'weight', 'ratings', 'x', 'y']
        cols_to_encode = [
            'position'
        ]
        scalers = {}

        for c in cols_to_normalize1 + cols_to_normalize2 + cols_to_encode:
            if self.METHOD == 'normal':
                if c in ['position_end', 'position_start']:
                    scalers[c] = {'mean':df_x2[c].mean(), 'std':df_x2[c].std()}
                else:
                    scalers[c] = {'mean':df_x[c].mean(), 'std':df_x[c].std()}
            if self.METHOD == 'min-max':
                if c in ['touch', 'sub', 
                        'Aerial', 'BallRecovery', 'BlockedPass', 'Card', 'Clearance', 'CornerAwarded', 'Dispossessed', 
                        'Foul', 'Interception', 'MissedShots', 'SavedShot', 'Tackle', 'TakeOn','OffsideProvoked']:
                    scalers[c] = {'min':df_x3[c].min(), 'max':df_x3[c].max()}
                else:
                    scalers[c] = {'min': df_x[c].min(), 'max': df_x[c].max()}

        scalers['pass'] = {'min' : min(df_x2['pass_cnts']), 'max' : max(df_x2['pass_cnts'])}

        return scalers, df_x, df_x2, df_x3, df_x3_home, df_x3_away, label_encoder, delete, cols_to_normalize1, cols_to_normalize2, cols_to_encode
    

class get_home_data:
    def __init__(self, df_x, df_x2, df_x3, xs_home, df_x3_home, edge_attributes_home, edge_indices_home, label_encoder, delete, results_home, cols_to_normalize1, cols_to_normalize2, cols_to_encode, scalers, device, METHOD = 'min-max'):
        self.df_x = df_x
        self.df_x2 = df_x2
        self.df_x3 = df_x3
        self.xs_home = xs_home 
        self.df_x3_home = df_x3_home 
        self.edge_attributes_home = edge_attributes_home 
        self.edge_indices_home = edge_indices_home
        self.label_encoder = label_encoder
        self.delete = delete
        self.results_home = results_home
        self.cols_to_normalize1 = cols_to_normalize1 
        self.cols_to_normalize2 = cols_to_normalize2 
        self.cols_to_encode = cols_to_encode
        self.METHOD = METHOD
        self.scalers = scalers
        self.device = device

    def get_dataset(self):
        dataset_new_home = []
        
        for i in range(len(self.xs_home)):
            if i not in self.delete:
                x_norm = self.xs_home[i]
                x_norm2 = [self.df_x3_home.loc[i, :].tolist()]
                edge_norm = self.edge_attributes_home[i]
                for n in range(len(x_norm)):
                    x_norm[n][0] = self.label_encoder.classes_.tolist().index(x_norm[n][0])


                x_norm = np.array(x_norm)
                x_norm2 = np.array(x_norm2)

                data = Data(x = torch.from_numpy(x_norm.astype(np.float32)), 
                            x2 = torch.from_numpy(x_norm2.astype(np.float32)),
                            edge_index = torch.from_numpy(self.edge_indices_home[i]),
                            edge_attr = torch.from_numpy(edge_norm.astype(np.int64)), 
                            y = torch.from_numpy(self.results_home[i].astype(np.float32)))
                edge_w_norm = edge_norm[:,0].astype(float)
                
                # normalize columns
                
                for c in self.cols_to_normalize2:
                    col_i = list(self.df_x.columns).index(c)

                    if self.METHOD == 'normal':
                        x_norm[:, col_i] = (x_norm[:, col_i] - self.scalers[c]['mean'])/self.scalers[c]['std']
                    if self.METHOD == 'min-max':
                        x_norm[:, col_i] = (x_norm[:, col_i] - self.scalers[c]['min'])/(self.scalers[c]['max'] - self.scalers[c]['min'])
                    
                for c in self.cols_to_normalize1:
                    col_i = list(self.df_x3.columns).index(c)

                    if self.METHOD == 'normal':
                        x_norm2[:, col_i] = (x_norm2[:, col_i] - self.scalers[c]['mean'])/self.scalers[c]['std']
                    if self.METHOD == 'min-max':
                        x_norm2[:, col_i] = (x_norm2[:, col_i] - self.scalers[c]['min'])/(self.scalers[c]['max'] - self.scalers[c]['min'])

                x_norm = x_norm.astype(np.float32)
                x_norm2 = x_norm2.astype(np.float32)

                edge_w_norm = (edge_w_norm - self.scalers['pass']['min'])/(self.scalers['pass']['max'] - self.scalers['pass']['min'])
                
                # saving results
                data.x_norm = torch.from_numpy(x_norm)
                data.x_norm2 = torch.from_numpy(x_norm2)
                data.edge_w_norm = torch.tensor(edge_w_norm, dtype=torch.float)
                
                dataset_new_home.append(data.to(self.device))
            else:
                pass

        return dataset_new_home
        
class get_away_data:
    def __init__(self, df_x, df_x2, df_x3, xs_away, df_x3_away, edge_attributes_away, edge_indices_away, label_encoder, delete, results_away, cols_to_normalize1, cols_to_normalize2, cols_to_encode, scalers, device, METHOD = 'min-max'):
        self.df_x = df_x
        self.df_x2 = df_x2
        self.df_x3 = df_x3
        self.xs_away = xs_away 
        self.df_x3_away = df_x3_away
        self.edge_attributes_away = edge_attributes_away 
        self.edge_indices_away = edge_indices_away
        self.label_encoder = label_encoder
        self.delete = delete
        self.results_away = results_away
        self.cols_to_normalize1 = cols_to_normalize1 
        self.cols_to_normalize2 = cols_to_normalize2 
        self.cols_to_encode = cols_to_encode
        self.METHOD = METHOD
        self.scalers = scalers
        self.device = device

    def get_dataset(self):
        dataset_new_away = []
        
        for i in range(len(self.xs_away)):
            if i not in self.delete:
                x_norm = self.xs_away[i]
                x_norm2 = [self.df_x3_away.loc[i, :].tolist()]
                edge_norm = self.edge_attributes_away[i]
                for n in range(len(x_norm)):
                    x_norm[n][0] = self.label_encoder.classes_.tolist().index(x_norm[n][0])


                x_norm = np.array(x_norm)
                x_norm2 = np.array(x_norm2)

                data = Data(x = torch.from_numpy(x_norm.astype(np.float32)), 
                            x2 = torch.from_numpy(x_norm2.astype(np.float32)),
                            edge_index = torch.from_numpy(self.edge_indices_away[i]),
                            edge_attr = torch.from_numpy(edge_norm.astype(np.int64)), 
                            y = torch.from_numpy(self.results_away[i].astype(np.float32)))
                edge_w_norm = edge_norm[:,0].astype(float)
                
                # normalize columns
                
                for c in self.cols_to_normalize2:
                    col_i = list(self.df_x.columns).index(c)

                    if self.METHOD == 'normal':
                        x_norm[:, col_i] = (x_norm[:, col_i] - self.scalers[c]['mean'])/self.scalers[c]['std']
                    if self.METHOD == 'min-max':
                        x_norm[:, col_i] = (x_norm[:, col_i] - self.scalers[c]['min'])/(self.scalers[c]['max'] - self.scalers[c]['min'])
                    
                for c in self.cols_to_normalize1:
                    col_i = list(self.df_x3.columns).index(c)

                    if self.METHOD == 'normal':
                        x_norm2[:, col_i] = (x_norm2[:, col_i] - self.scalers[c]['mean'])/self.scalers[c]['std']
                    if self.METHOD == 'min-max':
                        x_norm2[:, col_i] = (x_norm2[:, col_i] - self.scalers[c]['min'])/(self.scalers[c]['max'] - self.scalers[c]['min'])

                x_norm = x_norm.astype(np.float32)
                x_norm2 = x_norm2.astype(np.float32)

                edge_w_norm = (edge_w_norm - self.scalers['pass']['min'])/(self.scalers['pass']['max'] - self.scalers['pass']['min'])
                
                # saving results
                data.x_norm = torch.from_numpy(x_norm)
                data.x_norm2 = torch.from_numpy(x_norm2)
                data.edge_w_norm = torch.tensor(edge_w_norm, dtype=torch.float)
                
                dataset_new_away.append(data.to(self.device))
            else:
                pass

        return dataset_new_away


                        
