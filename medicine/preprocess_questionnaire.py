import numpy as np
import pandas as pd
import json

from preprocess_study_results import *

def preprocess_experience(x):
    try:
        return int(x)
    except:
        x = str(x)
        if x == 'nan':
            return np.nan
        else:
            return int(str(x)[:2])

def preprocess_confidence(x):
    try:
        return int(x)
    except:
        x = str(x)
        if x == 'nan':
            return np.nan
        elif '%' in x:
            return int("".join([i for i in x if i != '%']))
        elif '/' in x and 'edema' not in x:
            x0 = x.split('/')[0]
            x1 = x.split('/')[1]
            return float(x0) / float(x1)
        else:
            return np.nan

if __name__ == '__main__':

    # Load data
    data = pd.read_csv('data_raw_study_03.csv')

    # Load final data
    data_final = pd.read_csv('data_study_03.csv')

    # Choose final subset of data
    data_radio = data[data['ui'].isin(data_final['ui'])].copy()

    data_radio['treatment'] = 'None'
    data_radio['order'] = 'None'
     
    # Get column names to drop
    q_heat_forw = [i for i in data.columns if i.startswith('Q0') and i.endswith('F')]
    q_heat_back = [i for i in data.columns if i.startswith('Q0') and i.endswith('B')]
    q_no_heat_forw = [i for i in data.columns if i.startswith('Q1') and i.endswith('F')]
    q_no_heat_back = [i for i in data.columns if i.startswith('Q1') and i.endswith('B')]
    t_heat_forw = [i for i in data.columns if i.startswith('T0') and i.endswith('F_Page Submit')]
    t_heat_back = [i for i in data.columns if i.startswith('T0') and i.endswith('B_Page Submit')]
    t_no_heat_forw = [i for i in data.columns if i.startswith('T1') and i.endswith('F_Page Submit')]
    t_no_heat_back = [i for i in data.columns if i.startswith('T1') and i.endswith('B_Page Submit')]
    t_first_heat_forw = [i for i in data.columns if i.startswith('T0') and i.endswith('F_First Click')]
    t_first_heat_back = [i for i in data.columns if i.startswith('T0') and i.endswith('B_First Click')]
    t_first_no_heat_forw = [i for i in data.columns if i.startswith('T1') and i.endswith('F_First Click')]
    t_first_no_heat_back = [i for i in data.columns if i.startswith('T1') and i.endswith('B_First Click')]
    t_last_heat_forw = [i for i in data.columns if i.startswith('T0') and i.endswith('F_Last Click')]
    t_last_heat_back = [i for i in data.columns if i.startswith('T0') and i.endswith('B_Last Click')]
    t_last_no_heat_forw = [i for i in data.columns if i.startswith('T1') and i.endswith('F_Last Click')]
    t_last_no_heat_back = [i for i in data.columns if i.startswith('T1') and i.endswith('B_Last Click')]
    t_c_heat_forw = [i for i in data.columns if i.startswith('T0') and i.endswith('F_Click Count')]
    t_c_heat_back = [i for i in data.columns if i.startswith('T0') and i.endswith('B_Click Count')]
    t_c_no_heat_forw = [i for i in data.columns if i.startswith('T1') and i.endswith('F_Click Count')]
    t_c_no_heat_back = [i for i in data.columns if i.startswith('T1') and i.endswith('B_Click Count')]
    time_columns = [i for i in data.columns if i.startswith('Time ')]
     
    data_survey = data_radio.copy()
    data_survey = data_survey.drop(q_heat_forw, axis=1).copy()
    data_survey = data_survey.drop(q_heat_back, axis=1).copy()
    data_survey = data_survey.drop(q_no_heat_forw, axis=1).copy()
    data_survey = data_survey.drop(q_no_heat_back, axis=1).copy()
    data_survey = data_survey.drop(t_heat_forw, axis=1).copy()
    data_survey = data_survey.drop(t_heat_back, axis=1).copy()
    data_survey = data_survey.drop(t_no_heat_forw, axis=1).copy()
    data_survey = data_survey.drop(t_no_heat_back, axis=1).copy()
    data_survey = data_survey.drop(t_first_heat_forw, axis=1).copy()
    data_survey = data_survey.drop(t_first_heat_back, axis=1).copy()
    data_survey = data_survey.drop(t_first_no_heat_forw, axis=1).copy()
    data_survey = data_survey.drop(t_first_no_heat_back, axis=1).copy()
    data_survey = data_survey.drop(t_last_heat_forw, axis=1).copy()
    data_survey = data_survey.drop(t_last_heat_back, axis=1).copy()
    data_survey = data_survey.drop(t_last_no_heat_forw, axis=1).copy()
    data_survey = data_survey.drop(t_last_no_heat_back, axis=1).copy()
    data_survey = data_survey.drop(t_c_heat_forw, axis=1).copy()
    data_survey = data_survey.drop(t_c_heat_back, axis=1).copy()
    data_survey = data_survey.drop(t_c_no_heat_forw, axis=1).copy()
    data_survey = data_survey.drop(t_c_no_heat_back, axis=1).copy()
    data_survey = data_survey.drop(time_columns, axis=1).copy()
     
    data_survey = add_treatment_info(data_radio, data_survey, q_heat_forw, q_heat_back, q_no_heat_forw, q_no_heat_back)
     
    cog_load_columns = [i for i in data_survey.columns if i.startswith('Cognitive')]
    attitude_columns = [i for i in data_survey.columns if i.startswith('Attitude')]
    trust_columns = [i for i in data_survey.columns if i.startswith('Trust')]
     
    data_survey[cog_load_columns] = data_survey[cog_load_columns].replace(
        {'Very low': 1, 'Low': 2, 'Fairly Low': 3, 'Neutral': 4, 'Fairly high': 5, 'High': 6, 'Very high': 7})
    data_survey[attitude_columns] = data_survey[attitude_columns].replace(
        {'Extremely unlikely': 1, 'Quite unlikely': 2, 'Slightly unlikely': 3, 'Neither': 4, 'Slightly likely': 5,
         'Quite likely': 6, 'Extremely likely': 7})
    data_survey[trust_columns] = data_survey[trust_columns].replace(
        {'Strongly disagree': 1, 'Disagree': 2, 'Somewhat disagree': 3, 'Neutral': 4, 'Somewhat agree': 5, 'Agree': 6,
         'Strongly agree': 7})
     
    data_survey['Experience'] = data_survey['Experience'].apply(preprocess_experience)
     
    data_survey['IT skills_1'] = data_survey['IT skills_1'].replace(
        {'Novice': 1, 'Basic': 2, 'Good': 3, 'Very Good': 4, 'Expert': 5})
     
    data_survey['Radiology Exp_1'] = data_survey['Radiology Exp_1'].replace(
        {'Very low': 1, 'Low': 2, 'Fairly low': 3, 'Neutral': 4, 'Fairly high': 5, 'High': 6, 'Very high': 7})
     
    data_survey['Satisfaction_1'] = data_survey['Satisfaction_1'].replace(
        {'Completely dissatified': 1, 'Dissatisfied': 2, 'Somewhat dissatisfied': 3, 'Neutral': 4,
         'Somewhat satisfied': 5, 'Satisfied': 6, 'Completely satisfied': 7})
     
    data_survey['AI exp_1'] = data_survey['AI exp_1'].replace(
        {'Very little': 1, 'Little': 2, 'Some': 3, 'Much': 4, 'Very much': 5})
     
    data_survey['Exp AI_1'] = data_survey['Exp AI_1'].replace(
        {'Very poor': 1, 'Poor': 2, 'Fairly poor': 3, 'Neutral': 4, 'Fairly good': 5, 'Good': 6, 'Very good': 7})
     
    data_survey['Bad AI_1'] = data_survey['Bad AI_1'].replace(
        {'Very likely': 1, 'Likely': 2, 'Fairly likely': 3, 'Neutral': 4, 'Fairly unlikely': 5, 'Unlikely': 6,
         'Very unlikely': 7})
