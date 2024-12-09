import numpy as np
import pandas as pd
import json

from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score

# Define sensitivity and specificity
def sens_spec(true_labels, pred_labels):
    ## Get IDs with positive and negative true labels
    neg_ids = np.where(true_labels == 0)[0]
    pos_ids = np.where(true_labels == 1)[0]

    # Get sensitivity
    try:
        sens = (pred_labels[:, pos_ids] == 1).sum(axis=1) / len(pos_ids)
        spec = (pred_labels[:, neg_ids] == 0).sum(axis=1) / len(neg_ids)
    except:
        sens = (pred_labels[pos_ids] == 1).sum() / len(pos_ids)
        spec = (pred_labels[neg_ids] == 0).sum() / len(neg_ids)

    return sens, spec

def calc_adherence_correct(true_labels, pred_labels, ai_labels):
    correct_labels = true_labels == ai_labels
    adherence = (true_labels[correct_labels] == pred_labels[correct_labels]).sum() / correct_labels.sum()
    return adherence

def calc_overrule_wrong(true_labels, pred_labels, ai_labels):
    wrong_labels = true_labels != ai_labels
    overrule = (ai_labels[wrong_labels] != pred_labels[wrong_labels]).sum() / wrong_labels.sum()
    return overrule

def add_treatment_info(data, data_final, q_heat_forw, q_heat_back, q_no_heat_forw, q_no_heat_back):
    data = data.copy()
    data_final = data_final.copy()
    for ui in data_final.index:
        len_q_heat_forw = len(data.loc[ui, q_heat_forw].dropna())
        len_q_heat_back = len(data.loc[ui, q_heat_back].dropna())
        len_q_no_heat_forw = len(data.loc[ui, q_no_heat_forw].dropna())
        len_q_no_heat_back = len(data.loc[ui, q_no_heat_back].dropna())
        if len_q_heat_forw > 0:
            assert (len_q_heat_back == 0)
            assert (len_q_no_heat_forw == 0)
            assert (len_q_no_heat_back == 0)
            data_final.loc[ui, ['treatment', 'order',
                                'completed_images']] = 'human_with_explainable_AI', 'forward', len_q_heat_forw
        elif len_q_heat_back > 0:
            assert (len_q_heat_forw == 0)
            assert (len_q_no_heat_forw == 0)
            assert (len_q_no_heat_back == 0)
            data_final.loc[ui, ['treatment', 'order',
                          'completed_images']] = 'human_with_explainable_AI', 'backward', len_q_heat_back
        elif len_q_no_heat_forw > 0:
            assert (len_q_heat_forw == 0)
            assert (len_q_heat_back == 0)
            assert (len_q_no_heat_back == 0)
            data_final.loc[ui, ['treatment', 'order',
                                'completed_images']] = 'human_with_black-box_AI', 'forward', len_q_no_heat_forw
        elif len_q_no_heat_back > 0:
            assert (len_q_heat_forw == 0)
            assert (len_q_heat_back == 0)
            assert (len_q_no_heat_forw == 0)
            data_final.loc[ui, ['treatment', 'order',
                                'completed_images']] = 'human_with_black-box_AI', 'backward', len_q_no_heat_back
    return data_final

def is_in_us(lat, lon):
    # Bounding box for the continental United States
    continental_us = {
        'min_lat': 24.396308,
        'max_lat': 49.384358,
        'min_lon': -125.0,
        'max_lon': -66.93457
    }
    # Bounding box for Alaska
    alaska = {
        'min_lat': 51.209708,
        'max_lat': 71.538800,
        'min_lon': -179.148909,
        'max_lon': -129.979498
    }
    # Bounding box for Hawaii
    hawaii = {
        'min_lat': 18.910361,
        'max_lat': 28.402123,
        'min_lon': -178.334698,
        'max_lon': -154.806773
    }

    # Check if coordinates are within any of the bounding boxes
    def check_bounding_box(bbox):
        return bbox['min_lat'] <= lat <= bbox['max_lat'] and bbox['min_lon'] <= lon <= bbox['max_lon']

    return check_bounding_box(continental_us) or check_bounding_box(alaska) or check_bounding_box(hawaii)

# Function to check if a point is within Serbia's boundaries
def is_within_serbia(latitude, longitude):
    # Serbia's approximate geographical boundaries
    SERBIA_BOUNDARIES = {
        'north': 46.1900,
        'south': 42.2315,
        'west': 18.8378,
        'east': 23.0042,
    }
    return (SERBIA_BOUNDARIES['south'] <= latitude <= SERBIA_BOUNDARIES['north'] and
            SERBIA_BOUNDARIES['west'] <= longitude <= SERBIA_BOUNDARIES['east'])


if __name__ == '__main__':
    # Load results dictionary with true labels
    with open('results_dict_Lung Lesion.json', 'r') as fp:
        res = json.load(fp)

    # Get vector of true labels
    true_labels = []
    for key in res.keys():
        true_labels.append(res[key]['gt'])
    true_labels = np.array(true_labels)

    # Get AI performance
    ## Get AI predictions
    ai_labels = []
    for key in res.keys():
        if res[key]['score'] < 90:
            ai_labels.append(1)
        else:
            ai_labels.append(0)
    ai_labels = np.array(ai_labels)

    bacc = balanced_accuracy_score(true_labels, ai_labels)
    ddr = recall_score(true_labels, ai_labels)
    print(bacc, ddr)

    # Load data
    data = pd.read_csv('data_raw_study_03.csv')

    # Choose subset of data that was generated after publication
    old_len = len(data)
    data = data[data['Status'] == 'IP Address'].copy()
    new_len = len(data)
    print(f'Data from {old_len - new_len} participants were removed due to being generated before publication.')

    # Remove participants based in Serbia (used for testing by MSI)
    old_len = len(data)
    data = data[data.apply(lambda row: not is_within_serbia(float(row['LocationLatitude']),
                                                            float(row['LocationLongitude'])), axis=1)]
    new_len = len(data)
    print(f'Data from {old_len - new_len} participants were removed due to being in Serbia.')

    # Remove participants who did not complete study
    old_len = len(data)
    data = data[~data['Specialization'].isna()].copy()
    new_len = len(data)
    print(f'Data from {old_len - new_len} participants were removed due to not completing the study.')

    # Remove participants who are not in the US
    old_len = len(data)
    #data = data[data.apply(lambda x: is_in_us(float(x['LocationLatitude']), float(x['LocationLongitude'])),
    #                       axis=1)].copy()
    new_len = len(data)
    print(f'Data from {old_len - new_len} radiologists were removed due to location outside of US.')

    # Remove non-radiologists
    old_len = len(data)
    data = data[data['Specialization'] == 'Radiology'].copy()
    new_len = len(data)
    print(f'Data from {old_len - new_len} participants were removed due to not being radiologists.')

    # Remove participants who failed attention check in questionnaire
    old_len = len(data)
    #data = data[data['Attitude AI_7'] == 'Extremely likely'].copy()
    new_len = len(data)
    print(f'Data from {old_len - new_len} radiologists were removed due to failed attention check in questionnaire.')

    # Add ui as index
    data = data.set_index(data['ui']).drop('ui', axis=1)

    # Get column names
    q_heat_forw = [i for i in data.columns if i.startswith('Q0') and i.endswith('F')]
    q_heat_back = [i for i in data.columns if i.startswith('Q0') and i.endswith('B')]
    q_no_heat_forw = [i for i in data.columns if i.startswith('Q1') and i.endswith('F')]
    q_no_heat_back = [i for i in data.columns if i.startswith('Q1') and i.endswith('B')]
    t_heat_forw = [i for i in data.columns if i.startswith('T0') and i.endswith('F_Page Submit')]
    t_heat_back = [i for i in data.columns if i.startswith('T0') and i.endswith('B_Page Submit')]
    t_no_heat_forw = [i for i in data.columns if i.startswith('T1') and i.endswith('F_Page Submit')]
    t_no_heat_back = [i for i in data.columns if i.startswith('T1') and i.endswith('B_Page Submit')]

    # Set up final dataframe
    data_final = data[['IT skills_1', 'Experience', 'Specialization', 'LocationLongitude', 'LocationLatitude',
                       'Attitude AI_7']].copy()
    data_final = data_final.rename({'IT skills_1': 'it_skills', 'Experience': 'tenure',
                                    'Attitude AI_7': 'att_check'}, axis=1)
    data_final['tenure'] = data_final['tenure'].apply(lambda x: x[:2] if len(x) > 2 else x)
    data_final['tenure'] = data_final['tenure'].astype(int)
    data_final['it_skills'] = data_final['it_skills'].replace({'Novice': 1, 'Basic': 2, 'Good': 3,
                                                               'Very Good': 4, 'Expert': 5})
    data_final['att_check'] = data_final['att_check'].replace({'Extremely likely': 1, 'Quite likely': 0,
                                                               'Slightly likely': 0, 'Neither': 0, 'Quite unlikely': 0,
                                                               'Slightly unlikely': 0,
                                                               'Extremely unlikely': 0})
    data_final['treatment'] = 'None'
    data_final['order'] = 'None'
    data_final['completed_images'] = 0
    data_final['balanced_accuracy'] = np.nan
    data_final['defect_detection_rate'] = np.nan
    data_final['specificity'] = np.nan
    data_final['precision'] = np.nan
    data_final['adherence_accurate_prediction'] = np.nan
    data_final['overrule_wrong_prediction'] = np.nan
    data_final['median_decision_speed'] = np.nan
    data_final['only_one_answer'] = np.nan

    # Add treatment info
    data_final = add_treatment_info(data, data_final, q_heat_forw, q_heat_back, q_no_heat_forw, q_no_heat_back)

    for ui in data_final.index:
        if data_final.loc[ui, 'treatment'] == 'human_with_explainable_AI' and data_final.loc[ui, 'order'] == 'forward':
            pred_labels = np.array(data.loc[ui, q_heat_forw].replace({'NO': 0, 'YES': 1}))
            median_time = np.median(np.array(data.loc[ui, t_heat_forw].astype(float)))
        elif data_final.loc[ui, 'treatment'] == 'human_with_explainable_AI' and data_final.loc[
            ui, 'order'] == 'backward':
            pred_labels = np.array(data.loc[ui, q_heat_back].replace({'NO': 0, 'YES': 1}))
            pred_labels = np.flip(pred_labels)
            median_time = np.median(np.array(data.loc[ui, t_heat_back].astype(float)))
        elif data_final.loc[ui, 'treatment'] == 'human_with_black-box_AI' and data_final.loc[ui, 'order'] == 'forward':
            pred_labels = np.array(data.loc[ui, q_no_heat_forw].replace({'NO': 0, 'YES': 1}))
            median_time = np.median(np.array(data.loc[ui, t_no_heat_forw].astype(float)))
        elif data_final.loc[ui, 'treatment'] == 'human_with_black-box_AI' and data_final.loc[ui, 'order'] == 'backward':
            pred_labels = np.array(data.loc[ui, q_no_heat_back].replace({'NO': 0, 'YES': 1}))
            pred_labels = np.flip(pred_labels)
            median_time = np.median(np.array(data.loc[ui, t_no_heat_back].astype(float)))
        else:
            print(ui)
        data_final.loc[ui, 'balanced_accuracy'] = balanced_accuracy_score(true_labels, pred_labels)
        data_final.loc[ui, 'defect_detection_rate'] = recall_score(true_labels, pred_labels)
        data_final.loc[ui, 'specificity'] = sens_spec(true_labels, pred_labels)[1]
        data_final.loc[ui, 'precision'] = precision_score(true_labels, pred_labels)
        data_final.loc[ui, 'adherence_accurate_prediction'] = calc_adherence_correct(true_labels, pred_labels,
                                                                                     ai_labels)
        data_final.loc[ui, 'overrule_wrong_prediction'] = calc_overrule_wrong(true_labels, pred_labels, ai_labels)
        data_final.loc[ui, 'median_decision_speed'] = median_time
        data_final.loc[ui, 'only_one_answer'] = 1 if len(np.unique(pred_labels)) == 1 else 0

    # # Filter out participants who only submitted one answer for all tasks
    # old_len = len(data_final)
    # data_final = data_final[data_final['only_one_answer'] == 0].copy()
    # new_len = len(data_final)
    # print(f'Filtered out {old_len - new_len} participants who only submitted one answer for all tasks.')
    #
    # # Filter out participants whose performance is more than three standard deviations from the mean
    # # in their respective treatment group
    # old_len = len(data_final)
    # cutoff_heat = data_final["balanced_accuracy"][data_final["treatment"] == "human_with_explainable_AI"].mean() \
    #               - 3 * data_final["balanced_accuracy"][data_final["treatment"] == "human_with_explainable_AI"].std()
    # cutoff_no_heat = data_final["balanced_accuracy"][data_final["treatment"] == "human_with_black-box_AI"].mean() \
    #                  - 3 * data_final["balanced_accuracy"][data_final["treatment"] == "human_with_black-box_AI"].std()
    # data_final = data_final[~((data_final["treatment"] == "human_with_explainable_AI")
    #                           & (data_final["balanced_accuracy"] < cutoff_heat))]
    # data_final = data_final[~((data_final["treatment"] == "human_with_black-box_AI")
    #                           & (data_final["balanced_accuracy"] < cutoff_no_heat))]
    # new_len = len(data_final)
    # print(f'Filtered out {old_len - new_len} participants whose performance is more than three standard deviations'
    #       f' from the mean in their respective treatment group.')
    #
    # print(f'Final sample contains {len(data_final)} participants with '
    #       f'{len(data_final[data_final["treatment"] == "human_with_black-box_AI"])} in black-box AI treatment group and '
    #       f'{len(data_final[data_final["treatment"] == "human_with_explainable_AI"])} in explainable AI treatment group.')

    # Add participants that timed out for supplementary analysis
    data_raw = pd.read_csv('data_raw_study_03.csv')
    uis_time_out = ['56FF371F-DC4F-4890-9807-FC609CB267AA', 'F55DB32B-E919-4142-A91A-21C92E9B07C5']
    for ui in uis_time_out:
        data_sub = data_raw[data_raw['ui'] == ui].iloc[0].dropna().copy()
        questions = [i for i in data_sub.index if i.startswith('Q')]
        t_questions = [i for i in data_sub.index if i.startswith('T')]
        median_time = np.median(np.array(data_sub[t_questions].astype(float)))
        treatment = 'human_with_explainable_AI' if questions[0].startswith('Q0') else 'human_with_black-box_AI'
        order = 'forward' if questions[0].endswith('F') else 'backward'
        pred_labels = np.array(data_sub[questions].replace({'NO': 0, 'YES': 1}))
        if order == 'backward':
            pred_labels = np.flip(pred_labels)
        true_labels_incomplete = true_labels[:len(pred_labels)]
        data_final.loc[ui, 'balanced_accuracy'] = balanced_accuracy_score(true_labels_incomplete, pred_labels)
        data_final.loc[ui, 'defect_detection_rate'] = recall_score(true_labels_incomplete, pred_labels)
        data_final.loc[ui, 'specificity'] = sens_spec(true_labels_incomplete, pred_labels)[1]
        data_final.loc[ui, 'precision'] = precision_score(true_labels_incomplete, pred_labels)
        data_final.loc[ui, 'median_decision_speed'] = median_time
        data_final.loc[ui, 'only_one_answer'] = 1 if len(np.unique(pred_labels)) == 1 else 0
        data_final.loc[ui, 'treatment'] = treatment
        data_final.loc[ui, 'order'] = order
        data_final.loc[ui, 'completed_images'] = len(questions)

    data_final.to_csv('data_study_03.csv', index=True)