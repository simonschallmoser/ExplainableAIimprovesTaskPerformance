import numpy as np
import pandas as pd
import sqlite3

con = sqlite3.connect("db_aws_full.sqlite3")      

##### Preprocessing data #####

items = pd.read_sql_query("SELECT * from study_item", con)
items = items.rename(columns={"id": "item_id"})
users = pd.read_sql_query("SELECT * from landing_aws_surveyuser", con)
users = users.rename(columns={"user_hash": "user_id"})
users["user_id"] = users["user_id"].astype(str)
users = users[users["questionair_stage"] == 4]
users["start_study"] = pd.to_datetime(users["start_study"])
submissions = pd.read_sql_query("SELECT * from study_submission", con)
submissions["user_id"] = submissions["user_id"].astype(str)
submissions["submission_click"] = pd.to_datetime(submissions["submission_click"])
survey = pd.read_sql_query("SELECT * from survey_assessment", con)
survey["user_id"] = survey["user_id"].astype(str)
survey["age"] = abs(survey["age"])

submissions = submissions.merge(users.loc[:,["user_id", "study_group"]], on="user_id", how="left")
submissions = submissions.merge(items.loc[:,["item_id", "is_ok"]], on="item_id", how="left")
submissions = submissions[submissions["item_id"] > 10]
submissions = submissions.reset_index(drop=True)
submissions = submissions.sort_values(by="submission_click")

survey = survey.merge(users.loc[:,["user_id", "study_group"]], on="user_id", how="left")

submissions["false_nok"] = np.where((submissions["is_ok"] == 0) & (submissions["is_ok"] != submissions["final_selection"]), 1, 0)
submissions["false_ok"] = np.where((submissions["is_ok"] == 1) & (submissions["is_ok"] != submissions["final_selection"]), 1, 0)
submissions["time_delta"] = 0
for user in users["user_id"]:
    submissions["time_delta"][submissions["user_id"] == user] = submissions["submission_click"][submissions["user_id"] == user] - submissions["submission_click"][submissions["user_id"] == user].shift(1)
    submissions["time_delta"].loc[submissions["time_delta"][submissions["user_id"] == user].iloc[:1].index] = submissions["submission_click"][submissions["user_id"] == user].iloc[0] - users["start_study"][users["user_id"] == user].iloc[0]
submissions["time_delta"] = submissions["time_delta"].dt.total_seconds()
submissions = submissions.merge(items.loc[:,["item_id", "score"]], on = "item_id", how = "left")
submissions["ai_pred"] = np.where(submissions["score"] < 90, 0, 1)
submissions["adherence"] = np.where(((submissions["final_selection"] == 0) & (submissions["ai_pred"] == 0)) | ((submissions["final_selection"] == 1) & (submissions["ai_pred"] == 1)), 1, 0)
submissions["adherence_accurate_prediction"] = np.where(((submissions["adherence"] == 1) & (submissions["ai_pred"] == submissions["is_ok"])), 1, 0)
submissions["overrule"] = np.where(((submissions["final_selection"] == 0) & (submissions["ai_pred"] == 1)) | ((submissions["final_selection"] == 1) & (submissions["ai_pred"] == 0)), 1, 0)
submissions["overrule_wrong_prediction"] = np.where(((submissions["overrule"] == 1) & (submissions["ai_pred"] != submissions["is_ok"])), 1, 0)

survey.index = survey["user_id"]
survey["age"] = survey["age"]
survey["gender"] = survey["gender"]
survey["gender"] = np.where(survey["gender"] == "Female", 1, 0)
survey["education"][survey["education"] == "No schooling"] = 1
survey["education"][survey["education"] == "Primary school"] = 2
survey["education"][survey["education"] == "Some high school; no degree"] = 3
survey["education"][survey["education"] == "High school degree"] = 4
survey["education"][survey["education"] == "Bachelor's degree"] = 5
survey["education"][survey["education"] == "Master's degree"] = 6
survey["education"][survey["education"] == "Doctorate"] = 7
survey["age"][(survey["age"] >= 10) & (survey["age"] < 20)] = 1
survey["age"][(survey["age"] >= 20) & (survey["age"] < 30)] = 2
survey["age"][(survey["age"] >= 30) & (survey["age"] < 40)] = 3
survey["age"][(survey["age"] >= 40) & (survey["age"] < 50)] = 4
survey["age"][(survey["age"] >= 50) & (survey["age"] < 60)] = 5
survey["age"][(survey["age"] >= 60) & (survey["age"] < 70)] = 6
survey["age"][(survey["age"] >= 70) & (survey["age"] < 80)] = 7

participants = survey.loc[:,["user_id", "age", "gender", "education", "study_group"]]
participants.to_csv("participants.csv", index=False)

survey["it_skills"][survey["it_skills"] == "novice"] = 1
survey["it_skills"][survey["it_skills"] == "basic"] = 2
survey["it_skills"][survey["it_skills"] == "good"] = 3
survey["it_skills"][survey["it_skills"] == "very_good"] = 4
survey["it_skills"][survey["it_skills"] == "expert"] = 5
survey["ai_job_interaction"][survey["ai_job_interaction"] == "very_little"] = 1
survey["ai_job_interaction"][survey["ai_job_interaction"] == "little"] = 2
survey["ai_job_interaction"][survey["ai_job_interaction"] == "some"] = 3
survey["ai_job_interaction"][survey["ai_job_interaction"] == "much"] = 4
survey["ai_job_interaction"][survey["ai_job_interaction"] == "very_much"] = 5
survey["ai_familiarity"][survey["ai_familiarity"] == "very_little"] = 1
survey["ai_familiarity"][survey["ai_familiarity"] == "little"] = 2
survey["ai_familiarity"][survey["ai_familiarity"] == "some"] = 3
survey["ai_familiarity"][survey["ai_familiarity"] == "much"] = 4
survey["ai_familiarity"][survey["ai_familiarity"] == "very_much"] = 5

survey["performance_expectation"][survey["performance_expectation"] == "very_poor"] = 1
survey["performance_expectation"][survey["performance_expectation"] == "poor"] = 2
survey["performance_expectation"][survey["performance_expectation"] == "fairly_poor"] = 3
survey["performance_expectation"][survey["performance_expectation"] == "neutral"] = 4
survey["performance_expectation"][survey["performance_expectation"] == "fairly_good"] = 5
survey["performance_expectation"][survey["performance_expectation"] == "good"] = 6
survey["performance_expectation"][survey["performance_expectation"] == "very_good"] = 7
survey["perceived_error_sensitivity"][survey["perceived_error_sensitivity"] == "very_unlikely"] = 1
survey["perceived_error_sensitivity"][survey["perceived_error_sensitivity"] == "unlikely"] = 2
survey["perceived_error_sensitivity"][survey["perceived_error_sensitivity"] == "fairly_unlikely"] = 3
survey["perceived_error_sensitivity"][survey["perceived_error_sensitivity"] == "neutral"] = 4
survey["perceived_error_sensitivity"][survey["perceived_error_sensitivity"] == "fairly_likely"] = 5
survey["perceived_error_sensitivity"][survey["perceived_error_sensitivity"] == "likely"] = 6
survey["perceived_error_sensitivity"][survey["perceived_error_sensitivity"] == "very_likely"] = 7

survey["cognitive_load_1"][survey["cognitive_load_1"] == "very_low"] = 1
survey["cognitive_load_1"][survey["cognitive_load_1"] == "low"] = 2
survey["cognitive_load_1"][survey["cognitive_load_1"] == "fairly_low"] = 3
survey["cognitive_load_1"][survey["cognitive_load_1"] == "neutral"] = 4
survey["cognitive_load_1"][survey["cognitive_load_1"] == "fairly_high"] = 5
survey["cognitive_load_1"][survey["cognitive_load_1"] == "high"] = 6
survey["cognitive_load_1"][survey["cognitive_load_1"] == "very_high"] = 7
survey["cognitive_load_2"][survey["cognitive_load_2"] == "very_low"] = 1
survey["cognitive_load_2"][survey["cognitive_load_2"] == "low"] = 2
survey["cognitive_load_2"][survey["cognitive_load_2"] == "fairly_low"] = 3
survey["cognitive_load_2"][survey["cognitive_load_2"] == "neutral"] = 4
survey["cognitive_load_2"][survey["cognitive_load_2"] == "fairly_high"] = 5
survey["cognitive_load_2"][survey["cognitive_load_2"] == "high"] = 6
survey["cognitive_load_2"][survey["cognitive_load_2"] == "very_high"] = 7
survey["cognitive_load_3"][survey["cognitive_load_3"] == "very_low"] = 1
survey["cognitive_load_3"][survey["cognitive_load_3"] == "low"] = 2
survey["cognitive_load_3"][survey["cognitive_load_3"] == "fairly_low"] = 3
survey["cognitive_load_3"][survey["cognitive_load_3"] == "neutral"] = 4
survey["cognitive_load_3"][survey["cognitive_load_3"] == "fairly_high"] = 5
survey["cognitive_load_3"][survey["cognitive_load_3"] == "high"] = 6
survey["cognitive_load_3"][survey["cognitive_load_3"] == "very_high"] = 7
survey["cognitive_load_4"][survey["cognitive_load_4"] == "very_poor"] = 1
survey["cognitive_load_4"][survey["cognitive_load_4"] == "poor"] = 2
survey["cognitive_load_4"][survey["cognitive_load_4"] == "fairly_poor"] = 3
survey["cognitive_load_4"][survey["cognitive_load_4"] == "neutral"] = 4
survey["cognitive_load_4"][survey["cognitive_load_4"] == "fairly_good"] = 5
survey["cognitive_load_4"][survey["cognitive_load_4"] == "good"] = 6
survey["cognitive_load_4"][survey["cognitive_load_4"] == "very_good"] = 7
survey["cognitive_load_5"][survey["cognitive_load_5"] == "very_low"] = 1
survey["cognitive_load_5"][survey["cognitive_load_5"] == "low"] = 2
survey["cognitive_load_5"][survey["cognitive_load_5"] == "fairly_low"] = 3
survey["cognitive_load_5"][survey["cognitive_load_5"] == "neutral"] = 4
survey["cognitive_load_5"][survey["cognitive_load_5"] == "fairly_high"] = 5
survey["cognitive_load_5"][survey["cognitive_load_5"] == "high"] = 6
survey["cognitive_load_5"][survey["cognitive_load_5"] == "very_high"] = 7
survey["cognitive_load_6"][survey["cognitive_load_6"] == "very_low"] = 1
survey["cognitive_load_6"][survey["cognitive_load_6"] == "low"] = 2
survey["cognitive_load_6"][survey["cognitive_load_6"] == "fairly_low"] = 3
survey["cognitive_load_6"][survey["cognitive_load_6"] == "neutral"] = 4
survey["cognitive_load_6"][survey["cognitive_load_6"] == "fairly_high"] = 5
survey["cognitive_load_6"][survey["cognitive_load_6"] == "high"] = 6
survey["cognitive_load_6"][survey["cognitive_load_6"] == "very_high"] = 7

survey["perceived_usefulness_1"][survey["perceived_usefulness_1"] == "extremely_unlikely"] = 1
survey["perceived_usefulness_1"][survey["perceived_usefulness_1"] == "quite_unlikely"] = 2
survey["perceived_usefulness_1"][survey["perceived_usefulness_1"] == "slightly_unlikely"] = 3
survey["perceived_usefulness_1"][survey["perceived_usefulness_1"] == "neither"] = 4
survey["perceived_usefulness_1"][survey["perceived_usefulness_1"] == "slightly_likely"] = 5
survey["perceived_usefulness_1"][survey["perceived_usefulness_1"] == "quite_likely"] = 6
survey["perceived_usefulness_1"][survey["perceived_usefulness_1"] == "extremely_likely"] = 7
survey["perceived_usefulness_2"][survey["perceived_usefulness_2"] == "extremely_unlikely"] = 1
survey["perceived_usefulness_2"][survey["perceived_usefulness_2"] == "quite_unlikely"] = 2
survey["perceived_usefulness_2"][survey["perceived_usefulness_2"] == "slightly_unlikely"] = 3
survey["perceived_usefulness_2"][survey["perceived_usefulness_2"] == "neither"] = 4
survey["perceived_usefulness_2"][survey["perceived_usefulness_2"] == "slightly_likely"] = 5
survey["perceived_usefulness_2"][survey["perceived_usefulness_2"] == "quite_likely"] = 6
survey["perceived_usefulness_2"][survey["perceived_usefulness_2"] == "extremely_likely"] = 7
survey["perceived_usefulness_3"][survey["perceived_usefulness_3"] == "extremely_unlikely"] = 1
survey["perceived_usefulness_3"][survey["perceived_usefulness_3"] == "quite_unlikely"] = 2
survey["perceived_usefulness_3"][survey["perceived_usefulness_3"] == "slightly_unlikely"] = 3
survey["perceived_usefulness_3"][survey["perceived_usefulness_3"] == "neither"] = 4
survey["perceived_usefulness_3"][survey["perceived_usefulness_3"] == "slightly_likely"] = 5
survey["perceived_usefulness_3"][survey["perceived_usefulness_3"] == "quite_likely"] = 6
survey["perceived_usefulness_3"][survey["perceived_usefulness_3"] == "extremely_likely"] = 7
survey["perceived_usefulness_4"][survey["perceived_usefulness_4"] == "extremely_unlikely"] = 1
survey["perceived_usefulness_4"][survey["perceived_usefulness_4"] == "quite_unlikely"] = 2
survey["perceived_usefulness_4"][survey["perceived_usefulness_4"] == "slightly_unlikely"] = 3
survey["perceived_usefulness_4"][survey["perceived_usefulness_4"] == "neither"] = 4
survey["perceived_usefulness_4"][survey["perceived_usefulness_4"] == "slightly_likely"] = 5
survey["perceived_usefulness_4"][survey["perceived_usefulness_4"] == "quite_likely"] = 6
survey["perceived_usefulness_4"][survey["perceived_usefulness_4"] == "extremely_likely"] = 7
survey["perceived_usefulness_5"][survey["perceived_usefulness_5"] == "extremely_unlikely"] = 1
survey["perceived_usefulness_5"][survey["perceived_usefulness_5"] == "quite_unlikely"] = 2
survey["perceived_usefulness_5"][survey["perceived_usefulness_5"] == "slightly_unlikely"] = 3
survey["perceived_usefulness_5"][survey["perceived_usefulness_5"] == "neither"] = 4
survey["perceived_usefulness_5"][survey["perceived_usefulness_5"] == "slightly_likely"] = 5
survey["perceived_usefulness_5"][survey["perceived_usefulness_5"] == "quite_likely"] = 6
survey["perceived_usefulness_5"][survey["perceived_usefulness_5"] == "extremely_likely"] = 7
survey["perceived_usefulness_6"][survey["perceived_usefulness_6"] == "extremely_unlikely"] = 1
survey["perceived_usefulness_6"][survey["perceived_usefulness_6"] == "quite_unlikely"] = 2
survey["perceived_usefulness_6"][survey["perceived_usefulness_6"] == "slightly_unlikely"] = 3
survey["perceived_usefulness_6"][survey["perceived_usefulness_6"] == "neither"] = 4
survey["perceived_usefulness_6"][survey["perceived_usefulness_6"] == "slightly_likely"] = 5
survey["perceived_usefulness_6"][survey["perceived_usefulness_6"] == "quite_likely"] = 6
survey["perceived_usefulness_6"][survey["perceived_usefulness_6"] == "extremely_likely"] = 7

survey["perceived_ease_of_use_1"][survey["perceived_ease_of_use_1"] == "extremely_unlikely"] = 1
survey["perceived_ease_of_use_1"][survey["perceived_ease_of_use_1"] == "quite_unlikely"] = 2
survey["perceived_ease_of_use_1"][survey["perceived_ease_of_use_1"] == "slightly_unlikely"] = 3
survey["perceived_ease_of_use_1"][survey["perceived_ease_of_use_1"] == "neither"] = 4
survey["perceived_ease_of_use_1"][survey["perceived_ease_of_use_1"] == "slightly_likely"] = 5
survey["perceived_ease_of_use_1"][survey["perceived_ease_of_use_1"] == "quite_likely"] = 6
survey["perceived_ease_of_use_1"][survey["perceived_ease_of_use_1"] == "extremely_likely"] = 7
survey["perceived_ease_of_use_2"][survey["perceived_ease_of_use_2"] == "extremely_unlikely"] = 1
survey["perceived_ease_of_use_2"][survey["perceived_ease_of_use_2"] == "quite_unlikely"] = 2
survey["perceived_ease_of_use_2"][survey["perceived_ease_of_use_2"] == "slightly_unlikely"] = 3
survey["perceived_ease_of_use_2"][survey["perceived_ease_of_use_2"] == "neither"] = 4
survey["perceived_ease_of_use_2"][survey["perceived_ease_of_use_2"] == "slightly_likely"] = 5
survey["perceived_ease_of_use_2"][survey["perceived_ease_of_use_2"] == "quite_likely"] = 6
survey["perceived_ease_of_use_2"][survey["perceived_ease_of_use_2"] == "extremely_likely"] = 7
survey["perceived_ease_of_use_3"][survey["perceived_ease_of_use_3"] == "extremely_unlikely"] = 1
survey["perceived_ease_of_use_3"][survey["perceived_ease_of_use_3"] == "quite_unlikely"] = 2
survey["perceived_ease_of_use_3"][survey["perceived_ease_of_use_3"] == "slightly_unlikely"] = 3
survey["perceived_ease_of_use_3"][survey["perceived_ease_of_use_3"] == "neither"] = 4
survey["perceived_ease_of_use_3"][survey["perceived_ease_of_use_3"] == "slightly_likely"] = 5
survey["perceived_ease_of_use_3"][survey["perceived_ease_of_use_3"] == "quite_likely"] = 6
survey["perceived_ease_of_use_3"][survey["perceived_ease_of_use_3"] == "extremely_likely"] = 7
survey["perceived_ease_of_use_4"][survey["perceived_ease_of_use_4"] == "extremely_unlikely"] = 1
survey["perceived_ease_of_use_4"][survey["perceived_ease_of_use_4"] == "quite_unlikely"] = 2
survey["perceived_ease_of_use_4"][survey["perceived_ease_of_use_4"] == "slightly_unlikely"] = 3
survey["perceived_ease_of_use_4"][survey["perceived_ease_of_use_4"] == "neither"] = 4
survey["perceived_ease_of_use_4"][survey["perceived_ease_of_use_4"] == "slightly_likely"] = 5
survey["perceived_ease_of_use_4"][survey["perceived_ease_of_use_4"] == "quite_likely"] = 6
survey["perceived_ease_of_use_4"][survey["perceived_ease_of_use_4"] == "extremely_likely"] = 7
survey["perceived_ease_of_use_5"][survey["perceived_ease_of_use_5"] == "extremely_unlikely"] = 1
survey["perceived_ease_of_use_5"][survey["perceived_ease_of_use_5"] == "quite_unlikely"] = 2
survey["perceived_ease_of_use_5"][survey["perceived_ease_of_use_5"] == "slightly_unlikely"] = 3
survey["perceived_ease_of_use_5"][survey["perceived_ease_of_use_5"] == "neither"] = 4
survey["perceived_ease_of_use_5"][survey["perceived_ease_of_use_5"] == "slightly_likely"] = 5
survey["perceived_ease_of_use_5"][survey["perceived_ease_of_use_5"] == "quite_likely"] = 6
survey["perceived_ease_of_use_5"][survey["perceived_ease_of_use_5"] == "extremely_likely"] = 7
survey["perceived_ease_of_use_6"][survey["perceived_ease_of_use_6"] == "extremely_unlikely"] = 1
survey["perceived_ease_of_use_6"][survey["perceived_ease_of_use_6"] == "quite_unlikely"] = 2
survey["perceived_ease_of_use_6"][survey["perceived_ease_of_use_6"] == "slightly_unlikely"] = 3
survey["perceived_ease_of_use_6"][survey["perceived_ease_of_use_6"] == "neither"] = 4
survey["perceived_ease_of_use_6"][survey["perceived_ease_of_use_6"] == "slightly_likely"] = 5
survey["perceived_ease_of_use_6"][survey["perceived_ease_of_use_6"] == "quite_likely"] = 6
survey["perceived_ease_of_use_6"][survey["perceived_ease_of_use_6"] == "extremely_likely"] = 7

survey["ai_trust_1"][survey["ai_trust_1"] == "strongly_disagree"] = 1
survey["ai_trust_1"][survey["ai_trust_1"] == "disagree"] = 2
survey["ai_trust_1"][survey["ai_trust_1"] == "somewhat_disagree"] = 3
survey["ai_trust_1"][survey["ai_trust_1"] == "neither"] = 4
survey["ai_trust_1"][survey["ai_trust_1"] == "somewhat_agree"] = 5
survey["ai_trust_1"][survey["ai_trust_1"] == "agree"] = 6
survey["ai_trust_1"][survey["ai_trust_1"] == "strongly_agree"] = 7
survey["ai_trust_2"][survey["ai_trust_2"] == "strongly_disagree"] = 1
survey["ai_trust_2"][survey["ai_trust_2"] == "disagree"] = 2
survey["ai_trust_2"][survey["ai_trust_2"] == "somewhat_disagree"] = 3
survey["ai_trust_2"][survey["ai_trust_2"] == "neither"] = 4
survey["ai_trust_2"][survey["ai_trust_2"] == "somewhat_agree"] = 5
survey["ai_trust_2"][survey["ai_trust_2"] == "agree"] = 6
survey["ai_trust_2"][survey["ai_trust_2"] == "strongly_agree"] = 7
survey["ai_trust_3"][survey["ai_trust_3"] == "strongly_disagree"] = 1
survey["ai_trust_3"][survey["ai_trust_3"] == "disagree"] = 2
survey["ai_trust_3"][survey["ai_trust_3"] == "somewhat_disagree"] = 3
survey["ai_trust_3"][survey["ai_trust_3"] == "neither"] = 4
survey["ai_trust_3"][survey["ai_trust_3"] == "somewhat_agree"] = 5
survey["ai_trust_3"][survey["ai_trust_3"] == "agree"] = 6
survey["ai_trust_3"][survey["ai_trust_3"] == "strongly_agree"] = 7
survey["ai_trust_4"][survey["ai_trust_4"] == "strongly_disagree"] = 1
survey["ai_trust_4"][survey["ai_trust_4"] == "disagree"] = 2
survey["ai_trust_4"][survey["ai_trust_4"] == "somewhat_disagree"] = 3
survey["ai_trust_4"][survey["ai_trust_4"] == "neither"] = 4
survey["ai_trust_4"][survey["ai_trust_4"] == "somewhat_agree"] = 5
survey["ai_trust_4"][survey["ai_trust_4"] == "agree"] = 6
survey["ai_trust_4"][survey["ai_trust_4"] == "strongly_agree"] = 7
survey["ai_trust_5"][survey["ai_trust_5"] == "strongly_disagree"] = 1
survey["ai_trust_5"][survey["ai_trust_5"] == "disagree"] = 2
survey["ai_trust_5"][survey["ai_trust_5"] == "somewhat_disagree"] = 3
survey["ai_trust_5"][survey["ai_trust_5"] == "neither"] = 4
survey["ai_trust_5"][survey["ai_trust_5"] == "somewhat_agree"] = 5
survey["ai_trust_5"][survey["ai_trust_5"] == "agree"] = 6
survey["ai_trust_5"][survey["ai_trust_5"] == "strongly_agree"] = 7
survey["ai_trust_6"][survey["ai_trust_6"] == "strongly_disagree"] = 1
survey["ai_trust_6"][survey["ai_trust_6"] == "disagree"] = 2
survey["ai_trust_6"][survey["ai_trust_6"] == "somewhat_disagree"] = 3
survey["ai_trust_6"][survey["ai_trust_6"] == "neither"] = 4
survey["ai_trust_6"][survey["ai_trust_6"] == "somewhat_agree"] = 5
survey["ai_trust_6"][survey["ai_trust_6"] == "agree"] = 6
survey["ai_trust_6"][survey["ai_trust_6"] == "strongly_agree"] = 7
survey["ai_trust_7"][survey["ai_trust_7"] == "strongly_disagree"] = 1
survey["ai_trust_7"][survey["ai_trust_7"] == "disagree"] = 2
survey["ai_trust_7"][survey["ai_trust_7"] == "somewhat_disagree"] = 3
survey["ai_trust_7"][survey["ai_trust_7"] == "neither"] = 4
survey["ai_trust_7"][survey["ai_trust_7"] == "somewhat_agree"] = 5
survey["ai_trust_7"][survey["ai_trust_7"] == "agree"] = 6
survey["ai_trust_7"][survey["ai_trust_7"] == "strongly_agree"] = 7

survey["model_explained"][survey["model_explained"] == "strongly_disagree"] = 1
survey["model_explained"][survey["model_explained"] == "disagree"] = 2
survey["model_explained"][survey["model_explained"] == "somewhat_disagree"] = 3
survey["model_explained"][survey["model_explained"] == "neither"] = 4
survey["model_explained"][survey["model_explained"] == "somewhat_agree"] = 5
survey["model_explained"][survey["model_explained"] == "agree"] = 6
survey["model_explained"][survey["model_explained"] == "strongly_agree"] = 7
survey["relied_upon"][survey["relied_upon"] == "very_little"] = 1
survey["relied_upon"][survey["relied_upon"] == "little"] = 2
survey["relied_upon"][survey["relied_upon"] == "fairly_little"] = 3
survey["relied_upon"][survey["relied_upon"] == "neutral"] = 4
survey["relied_upon"][survey["relied_upon"] == "fairly_much"] = 5
survey["relied_upon"][survey["relied_upon"] == "much"] = 6
survey["relied_upon"][survey["relied_upon"] == "very_much"] = 7

survey_columns = ["user_id",
                  "it_skills",
                  "ai_job_interaction",
                  "ai_familiarity",
                  "confidence_own",
                  "performance_expectation",
                  "perceived_error_sensitivity",
                  "cognitive_load_1",
                  "cognitive_load_2",
                  "cognitive_load_3",
                  "cognitive_load_4",
                  "cognitive_load_5",
                  "cognitive_load_6",
                  "perceived_usefulness_1",
                  "perceived_usefulness_2",
                  "perceived_usefulness_3",
                  "perceived_usefulness_4",
                  "perceived_usefulness_5",
                  "perceived_usefulness_6",
                  "perceived_ease_of_use_1",
                  "perceived_ease_of_use_2",
                  "perceived_ease_of_use_3",
                  "perceived_ease_of_use_4",
                  "perceived_ease_of_use_5",
                  "perceived_ease_of_use_6",
                  "ai_trust_1",
                  "ai_trust_2",
                  "ai_trust_3",
                  "ai_trust_4",
                  "ai_trust_5",
                  "ai_trust_6",
                  "ai_trust_7",
                  "model_explained",
                  "relied_upon"]

survey = survey.loc[:, survey_columns]
survey.to_csv("survey.csv", index=False)

summary = users.loc[:, ["user_id"]]
summary.index = users["user_id"]
summary["total_duration"] = submissions.loc[:,["user_id", "time_delta"]].groupby("user_id").sum()
summary["median_duration"] = submissions.loc[:,["user_id", "time_delta"]].groupby("user_id").median()
summary["seen"] = submissions.loc[:,["user_id", "is_ok"]].groupby("user_id").count()
summary["seen_nok"] = (submissions.loc[:,["user_id", "is_ok"]].groupby("user_id").count() - submissions.loc[:,["user_id", "is_ok"]].groupby("user_id").sum())
summary["seen_ok"] = submissions.loc[:,["user_id", "is_ok"]].groupby("user_id").sum()
summary["false_nok"] = submissions.loc[:,["user_id", "false_nok"]].groupby("user_id").sum()
summary["false_ok"] = submissions.loc[:,["user_id", "false_ok"]].groupby("user_id").sum()
summary["correct_nok"] = summary["seen_nok"] - summary["false_nok"]
summary["correct_ok"] = summary["seen_ok"] - summary["false_ok"]
summary["defect_detection_rate"] = summary["correct_nok"]/summary["seen_nok"]
summary["balanced_accuracy"] = 0.5*(summary["correct_ok"]/summary["seen_ok"] + summary["correct_nok"]/summary["seen_nok"])
summary["accuracy"] = (summary["correct_ok"] + summary["correct_nok"])/summary["seen"]
summary["adherence_accurate_prediction"] = submissions.loc[:,["user_id", "adherence_accurate_prediction"]].groupby("user_id").sum()/submissions[submissions["ai_pred"] == submissions["is_ok"]].loc[:,["user_id", "adherence_accurate_prediction"]].groupby("user_id").count()
summary["overrule_wrong_prediction"] = submissions.loc[:,["user_id", "overrule_wrong_prediction"]].groupby("user_id").sum()/submissions[submissions["ai_pred"] != submissions["is_ok"]].loc[:,["user_id", "overrule_wrong_prediction"]].groupby("user_id").count()

summary_columns = ["user_id",
                   "total_duration",
                   "median_duration",
                   "seen",
                   "defect_detection_rate",
                   "balanced_accuracy",
                   "accuracy",
                   "adherence_accurate_prediction",
                   "overrule_wrong_prediction"]
summary = summary.loc[:, summary_columns]
summary.to_csv("summary.csv", index=False)

##### Published data #####

participants = pd.read_csv("participants.csv")
survey = pd.read_csv("survey.csv")
summary = pd.read_csv("summary.csv")

data = participants.merge(survey, on="user_id", how="left")
data = data.merge(summary, on="user_id", how="left")
data = data.loc[:,["study_group", "seen", "balanced_accuracy", "defect_detection_rate", "adherence_accurate_prediction", "overrule_wrong_prediction", "median_duration", "it_skills", "age", "education", "gender"]]

data["study_group"][data["study_group"] == 0] = "human_without_AI"
data["study_group"][data["study_group"] == 1] = "human_with_black-box_AI"
data["study_group"][data["study_group"] == 2] = "human_with_explainable_AI"

columns = ["treatment", "completed_images", "balanced_accuracy", "defect_detection_rate", "adherence_accurate_prediction", "overrule_wrong_prediction", "median_decision_speed", "it_skills", "age", "education", "gender"]
data.columns = columns
data.to_csv("data_study_01.csv", index=False)
