
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import time
from datetime import datetime
from Wisard import Wisard


# In[2]:


# The preprocessing part is partialy based on the following "kernel" on Kaggle:
# https://www.kaggle.com/jgoldberg/donorschoose-eda-text-classification/notebook

def preprocess(training_dataframe, resources_dataframe):
    #print(training_dataframe.shape)
    #print(resources_dataframe.shape)
    
    #
    total_price = resources_dataframe.quantity * resources_dataframe.price
    resources_dataframe["total_price"] = total_price
    
    # dropping irrelevant columns
    resources_dataframe = resources_dataframe.drop(["description", "price"], axis=1)
    training_dataframe = training_dataframe.drop(["teacher_id"], axis=1)
    
    # grouping resources data by id
    grouped_resources_dataframe = resources_dataframe.groupby("id", as_index=False, sort=False).sum()
    grouped_resources_dataframe
    
    # merging the two dataframes
    cleaned_df = pd.merge(training_dataframe, grouped_resources_dataframe, how="inner", on=["id"])
    
    # splitting project categories
    
    cleaned_df[['category_1','category_2']] = cleaned_df['project_subject_categories'].str.replace(", Care & Hunger", "").str.split(', ', 3, expand=True)
    #print(cleaned_df['project_subject_categories'].str.replace(", Care & Hunger", "").str.split(', ', 3, expand=True))
    
    cleaned_df[['subcategory_1','subcategory_2']] = cleaned_df['project_subject_subcategories'].str.replace(", Care & Hunger", "").str.split(', ', 3, expand=True)
    
    #cleaned_df["category_1"] = cleaned_df["category_1"].fillna("Not Informed")
    cleaned_df["category_2"] = cleaned_df["category_2"].fillna("Not Informed")
    cleaned_df["subcategory_2"] = cleaned_df["subcategory_2"].fillna("Not Informed")
    
    cleaned_df["total_price_category"] = pd.cut(
        cleaned_df["total_price"], 
        bins=[0,100,250,500,1000,16000], 
        labels=["0-100","101-250","251-500","501-1000",">1000"]
    )
    
    cleaned_df["n_previous_projects"] = pd.cut(
        cleaned_df["teacher_number_of_previously_posted_projects"],
        bins=[-1,1,5,10,25,50,500],
        labels=['0-1','2-5','6-10','11-25','26-50','51+']
    )
    
    cleaned_df["project_submitted_datetime"] = pd.to_datetime(cleaned_df['project_submitted_datetime'])
    cleaned_df["month"] = cleaned_df['project_submitted_datetime'].dt.month
    cleaned_df["quarter"] = cleaned_df['project_submitted_datetime'].dt.quarter
    
    cleaned_df["teacher_prefix"] = cleaned_df["teacher_prefix"].fillna("unknown")
    
    cleaned_df["project_essay_1"] = cleaned_df["project_essay_1"].fillna("")
    cleaned_df["project_essay_2"] = cleaned_df["project_essay_2"].fillna("")
    cleaned_df["project_essay_3"] = cleaned_df["project_essay_3"].fillna("")
    cleaned_df["project_essay_4"] = cleaned_df["project_essay_4"].fillna("")
    
    #cleaned_df["merged_essays"] = cleaned_df['project_title'].astype(str) + " " + cleaned_df['project_essay_1'].astype(str) + " " + cleaned_df['project_essay_2'].astype(str) + " " + cleaned_df['project_essay_3'].astype(str) + " " + cleaned_df['project_essay_4'].astype(str)
    cleaned_df["merged_essays"] = cleaned_df['project_essay_1'].astype(str) + " " + cleaned_df['project_essay_2'].astype(str) + " " + cleaned_df['project_essay_3'].astype(str) + " " + cleaned_df['project_essay_4'].astype(str)
    
    # dropping more columns
    cleaned_df = cleaned_df.drop([
        "project_submitted_datetime", 
        "project_essay_1", 
        "project_essay_2", 
        "project_essay_3", 
        "project_essay_4",
        "quantity",
        "total_price",
        "teacher_number_of_previously_posted_projects"], 
        axis=1
    )
    
    return cleaned_df

# returns a list with the following format
# [
#     ["001101...010101", 1]
#     ["001111...000001", 1]
#     ["101001...111100", 0]
# ]
def convert_to_bits_string(dataframe):
    print("Converting dataframe of shape ", dataframe.shape, " to a list of binary values.")
    
    project_grade_category_mapping = { # 4
        'Grades PreK-2':  "1000", 
        'Grades 3-5':     "1100", 
        'Grades 6-8':     "1110", 
        'Grades 9-12':    "1111"
    }

    teacher_prefix_mapping = { # 6
        'Ms.':     "100000", 
        'Mrs.':    "110000", 
        'Mr.':     "111000", 
        'Teacher': "111100", 
        'Dr.':     "111110", 
        'unknown': "111111"
    }

    n_previous_projects_mapping = { # 6
        '0-1':     "100000",
        '2-5':     "110000",
        '6-10':    "111000",
        '11-25':   "111100",
        '26-50':   "111110",
        '51+':     "111111"
    }

    total_price_category_mapping = { # 6
        "0-100":     "100000",
        "101-250":   "110000",
        "251-500":   "111000",
        "501-1000":  "111100",
        ">1000":     "111110"
    }
    
    month_mapping = { # 12
        "1":  "100000000000",
        "2":  "110000000000",
        "3":  "111000000000",
        "4":  "111100000000",
        "5":  "111110000000",
        "6":  "111111000000",
        "7":  "111111100000",
        "8":  "111111110000",
        "9":  "111111111000",
        "10": "111111111100",
        "11": "111111111110",
        "12": "111111111111"
    }
    
    quarter_mapping = { # 4
        "1":"1000",
        "2":"1100",
        "3":"1110",
        "4":"1111"
    }
    
    category_mapping = { # 10
        "Not Informed":          "1000000000",
        "Applied Learning":      "1100000000",
        "Health & Sports":       "1110000000",
        "History & Civics":      "1111000000",
        "Literacy & Language":   "1111100000",
        "Math & Science":        "1111110000",
        "Music & The Arts":      "1111111000",
        "Special Needs":         "1111111100",
        "Warmth":                "1111111110",
        "Care & Hunger":         "1111111110", # Equals to warmth, because they are the same thing
    }
    
    subcategory_mapping = { # 30
        "Not Informed":          "100000000000000000000000000000",
        "Literacy":              "110000000000000000000000000000",
        "Performing Arts":       "111000000000000000000000000000",
        "Applied Sciences":      "111100000000000000000000000000",
        "Health & Wellness":     "111110000000000000000000000000",
        "Character Education":   "111111000000000000000000000000",
        "Early Development":     "111111100000000000000000000000",
        "Mathematics":           "111111110000000000000000000000",
        "Literature & Writing":  "111111111000000000000000000000",
        "Special Needs":         "111111111100000000000000000000", 
        "ESL":                   "111111111110000000000000000000", 
        "Health & Life Science": "111111111111000000000000000000", 
        "College & Career Prep": "111111111111100000000000000000", 
        "Environmental Science": "111111111111110000000000000000", 
        "Other":                 "111111111111111000000000000000", 
        "Music":                 "111111111111111100000000000000", 
        "Visual Arts":           "111111111111111110000000000000", 
        "History & Geography":   "111111111111111111000000000000", 
        "Gym & Fitness":         "111111111111111111100000000000", 
        "Warmth":                "111111111111111111110000000000", 
        "Extracurricular":       "111111111111111111111000000000", 
        "Team Sports":           "111111111111111111111100000000", 
        "Social Sciences":       "111111111111111111111110000000", 
        "Foreign Languages":     "111111111111111111111111000000", 
        "Parent Involvement":    "111111111111111111111111100000", 
        "Nutrition Education":   "111111111111111111111111110000", 
        "Community Service":     "111111111111111111111111111000", 
        "Financial Literacy":    "111111111111111111111111111100", 
        "Civics & Government":   "111111111111111111111111111110", 
        "Economics":             "111111111111111111111111111111", 
        
    }
    
    school_state_mapping = { # 50
        'NV':"10000000000000000000000000000000000000000000000000", 
        'GA':"11000000000000000000000000000000000000000000000000", 
        'UT':"11100000000000000000000000000000000000000000000000", 
        'NC':"11110000000000000000000000000000000000000000000000", 
        'CA':"11111000000000000000000000000000000000000000000000", 
        'DE':"11111100000000000000000000000000000000000000000000", 
        'MO':"11111110000000000000000000000000000000000000000000", 
        'SC':"11111111000000000000000000000000000000000000000000", 
        'IN':"11111111100000000000000000000000000000000000000000", 
        'IL':"11111111110000000000000000000000000000000000000000", 
        'VA':"11111111111000000000000000000000000000000000000000",
        'PA':"11111111111100000000000000000000000000000000000000", 
        'NY':"11111111111110000000000000000000000000000000000000", 
        'FL':"11111111111111000000000000000000000000000000000000", 
        'NJ':"11111111111111100000000000000000000000000000000000", 
        'TX':"11111111111111110000000000000000000000000000000000", 
        'LA':"11111111111111111000000000000000000000000000000000", 
        'ID':"11111111111111111100000000000000000000000000000000", 
        'OH':"11111111111111111110000000000000000000000000000000", 
        'OR':"11111111111111111111000000000000000000000000000000", 
        'MD':"11111111111111111111100000000000000000000000000000", 
        'WA':"11111111111111111111110000000000000000000000000000",
        'MA':"11111111111111111111111000000000000000000000000000", 
        'KY':"11111111111111111111111100000000000000000000000000", 
        'AZ':"11111111111111111111111110000000000000000000000000", 
        'MI':"11111111111111111111111111000000000000000000000000", 
        'CT':"11111111111111111111111111100000000000000000000000", 
        'AR':"11111111111111111111111111110000000000000000000000", 
        'WV':"11111111111111111111111111111000000000000000000000", 
        'NM':"11111111111111111111111111111100000000000000000000", 
        'WI':"11111111111111111111111111111110000000000000000000", 
        'MN':"11111111111111111111111111111111000000000000000000", 
        'OK':"11111111111111111111111111111111100000000000000000",
        'AL':"11111111111111111111111111111111110000000000000000", 
        'TN':"11111111111111111111111111111111111000000000000000", 
        'IA':"11111111111111111111111111111111111100000000000000", 
        'KS':"11111111111111111111111111111111111110000000000000", 
        'CO':"11111111111111111111111111111111111111000000000000", 
        'DC':"11111111111111111111111111111111111111100000000000", 
        'WY':"11111111111111111111111111111111111111110000000000", 
        'NH':"11111111111111111111111111111111111111111000000000", 
        'HI':"11111111111111111111111111111111111111111100000000", 
        'SD':"11111111111111111111111111111111111111111110000000", 
        'MT':"11111111111111111111111111111111111111111111000000",
        'MS':"11111111111111111111111111111111111111111111100000", 
        'RI':"11111111111111111111111111111111111111111111110000", 
        'VT':"11111111111111111111111111111111111111111111111000", 
        'ME':"11111111111111111111111111111111111111111111111100", 
        'NE':"11111111111111111111111111111111111111111111111110", 
        'AK':"11111111111111111111111111111111111111111111111111", 
        'ND':"00000000000000000000000000000000000000000000000000"
    }
    
    combined_input_and_expected_output = []
    input_list = []
    expected_output_list = []
    
    n = 0
    for index, row in dataframe.iterrows():
        #print(row)
        #if n >= 10:
        #    break
        #n = n + 1
        
        # total 128 bits
        bits_string = ""
        bits_string = project_grade_category_mapping[row["project_grade_category"]]
        bits_string = bits_string + teacher_prefix_mapping[row["teacher_prefix"]]
        bits_string = bits_string + n_previous_projects_mapping[row["n_previous_projects"]]
        bits_string = bits_string + total_price_category_mapping[row["total_price_category"]]
        
        bits_string = bits_string + month_mapping[str(row["month"])]
        bits_string = bits_string + quarter_mapping[str(row["quarter"])]
        bits_string = bits_string + category_mapping[row["category_1"]]
        bits_string = bits_string + category_mapping[row["category_2"]] # perhaps it is possible to ignore this one
        
        bits_string = bits_string + subcategory_mapping[row["subcategory_1"]]
        bits_string = bits_string + subcategory_mapping[row["subcategory_2"]]
        bits_string = bits_string + school_state_mapping[row["school_state"]]
        
        bit_int_list = [int(c) for c in bits_string]
        expected_output = str(row["project_is_approved"])
        
        input_list.append(bit_int_list)
        expected_output_list.append(expected_output)
        
        combined_input_and_expected_output.append([bit_int_list, expected_output])
        
    return input_list, expected_output_list, combined_input_and_expected_output


# In[3]:


def loadData():
    # load data
    train_file_path = 'train.csv'
    # Test data file not considered because it doesn't contain the classes of the entries
    #test_file_path = 'test.csv'
    resources_file_path = 'resources.csv'

    # Read data and store in DataFrame
    train_data = pd.read_csv(train_file_path, sep=',')
    #test_data = pd.read_csv(test_file_path, sep=',')
    resources_data = pd.read_csv(resources_file_path, sep=',')
    
    return train_data, resources_data

# splitting the training dataset into training and test, because the official test dataset
# doesn't have the entries' classification, requiring validation with Kaggle's website
def splitData(train_data, resources_data, training_set_total_aproved, training_set_total_reproved):
    
    print("Total data: ", len(train_data))
    print("Total aproved: ", train_data["project_is_approved"].sum())
    print("Total reproved: ", len(train_data) - train_data["project_is_approved"].sum())
    print("Percent aproved: ", float(train_data["project_is_approved"].sum()) / float(len(train_data)))
    print("Percent reproved: ", 1.0 - (float(train_data["project_is_approved"].sum()) / float(len(train_data))), "\n")

    #train = train_data.sample(n=10000,random_state=200)
    train = train_data.sample(frac=1,random_state=200)
    print("Distribution over a random sample of 182080 observations used to get the observations to train the classifier: ",
          float(train["project_is_approved"].sum()) / float(len(train["project_is_approved"])))
    print("Total aproved in that sample: ", train["project_is_approved"].sum(), "\n")
    
    aproved = train[train["project_is_approved"] == 1][:training_set_total_aproved]
    reproved = train[train["project_is_approved"] == 0][:training_set_total_reproved]

    training_set = pd.concat([aproved, reproved])
    training_set = training_set.sample(frac=1, random_state=200)
    test_set = train_data.drop(training_set.index)

    print("Total training data: ", len(training_set))
    print("Total aproved: ", training_set["project_is_approved"].sum())
    print("Total reproved: ", len(training_set) - training_set["project_is_approved"].sum())
    print("Percent aproved: ", float(training_set["project_is_approved"].sum()) / float(len(training_set)))
    print("Percent reproved: ", 1.0 - (float(training_set["project_is_approved"].sum()) / float(len(training_set))), "\n")

    print("Total test data: ", len(test_set))
    print("Total aproved: ", test_set["project_is_approved"].sum())
    print("Total reproved: ", len(test_set) - test_set["project_is_approved"].sum())
    print("Percent aproved: ", float(test_set["project_is_approved"].sum()) / float(len(test_set)))
    print("Percent reproved: ", 1.0 - (float(test_set["project_is_approved"].sum()) / float(len(test_set))), "\n")

    print("Training set + test set: ", len(training_set) + len(test_set))

    return training_set, test_set


def getData(training_set_total_aproved, training_set_total_reproved):
    train_data, resources_data = loadData()
    training_set, test_set = splitData(train_data, resources_data, training_set_total_aproved, training_set_total_reproved)
    
    training_df = preprocess(training_set, resources_data)
    test_df = preprocess(test_set, resources_data)
    
    training_input, expected_output, training_combined = convert_to_bits_string(training_df)
    test_input, test_expected_output, test_combined = convert_to_bits_string(test_df)
    
    return training_input, expected_output, training_combined, test_input, test_expected_output, test_combined, training_set, test_set


# In[4]:


# Trains using a WiSARD classifier
# Using personal implementation, without bleaching
def train(training_input, expected_output, tuple_size = 2, bleaching = False):
    wann = Wisard(tuple_size, 3546, bleaching)
    wann.train(training_input, expected_output)
    return wann


# In[5]:


#Evaluates Guilherme's wisard implementation
def evaluate_performance(wann, test_data_combined):
    #print("Number of observations: ", test_data_combined)
    correct_predictions = 0
    wrong_predictions = 0
    zeros_predicted = 0
    ones_predicted = 0
    zeros_correct = 0
    ones_correct = 0
    zeros_wrong = 0
    ones_wrong = 0
    total_ties = 0
    ones_total_ties = 0
    zeros_total_ties = 0
    avg_time = time.time()
    
    for combined in test_data_combined:
        prediction, tie = wann.predict(combined[0])
        prediction = prediction["class"]
        
        if prediction == "0":
            #print("Prediction: ", prediction[0], combined)
            zeros_predicted = zeros_predicted + 1
        elif prediction == "1":
            ones_predicted = ones_predicted + 1
        #print(prediction)
        
        expected = combined[1]
        #print(prediction, expected)
        if prediction == expected:
            #print("Correct!")
            correct_predictions = correct_predictions + 1
            
            if prediction == "0":
                zeros_correct = zeros_correct + 1
            elif prediction == "1":
                ones_correct = ones_correct + 1
        else:
            wrong_predictions = wrong_predictions + 1
            
            if prediction == "0":
                zeros_wrong = zeros_wrong + 1
            elif prediction == "1":
                ones_wrong = ones_wrong + 1
                
        if tie:
            total_ties = total_ties + 1
            if expected == "0":
                zeros_total_ties = zeros_total_ties + 1
            elif expected == "1":
                ones_total_ties = ones_total_ties + 1
                
    avg_time = float(time.time() - avg_time) / float(len(test_data_combined)) 
    
    print("Number of observations: ", len(test_data_combined))
    print("Predicted correctly: ", correct_predictions)
    print("Predicted wrongly: ", wrong_predictions)
    print("Predicted zeros: ", zeros_predicted)
    print("Predicted ones: ", ones_predicted)
    print("Zeros correct: ", zeros_correct)
    print("Ones correct: ", ones_correct)
    print("Zeros wrong: ", zeros_wrong)
    print("Ones Wrong: ", ones_wrong)
    print("Total ties: ", total_ties)
    print("Zeros total ties: ", zeros_total_ties)
    print("Ones total ties: ", ones_total_ties)
    print("Avg. Time: ", avg_time, " seconds.")
    return correct_predictions, [
        len(test_data_combined), correct_predictions, wrong_predictions, zeros_predicted, ones_predicted,
        zeros_correct, ones_correct, zeros_wrong, ones_wrong, total_ties, zeros_total_ties, ones_total_ties,
        avg_time
    ]

#Evaluates Firminos's wisard implementation
def evaluate_performance2(w, test_data_combined):
    #print("Number of observations: ", test_data_combined)
    correct_predictions = 0
    wrong_predictions = 0
    zeros_predicted = 0
    ones_predicted = 0
    zeros_correct = 0
    ones_correct = 0
    zeros_wrong = 0
    ones_wrong = 0
    avg_time = time.time()
    
    for combined in test_data_combined:
        prediction = w.predict([combined[0]])
        if prediction[0] == "0":
            #print("Prediction: ", prediction[0], combined)
            zeros_predicted = zeros_predicted + 1
        elif prediction[0] == "1":
            ones_predicted = ones_predicted + 1
        #print(prediction)
        expected = combined[1]
        #print(prediction, expected)
        if prediction[0] == expected:
            #print("Correct!")
            correct_predictions = correct_predictions + 1
            
            if prediction[0] == "0":
                zeros_correct = zeros_correct + 1
            elif prediction[0] == "1":
                ones_correct = ones_correct + 1
        else:
            wrong_predictions = wrong_predictions + 1
            
            if prediction[0] == "0":
                zeros_wrong = zeros_wrong + 1
            elif prediction[0] == "1":
                ones_wrong = ones_wrong + 1
                
    avg_time = float(time.time() - avg_time) / float(len(test_data_combined)) 
    
    print("Number of observations: ", len(test_data_combined))
    print("Predicted correctly: ", correct_predictions)
    print("Predicted wrongly: ", wrong_predictions)
    print("Predicted zeros: ", zeros_predicted)
    print("Predicted ones: ", ones_predicted)
    print("Zeros correct: ", zeros_correct)
    print("Ones correct: ", ones_correct)
    print("Zeros wrong: ", zeros_wrong)
    print("Ones Wrong: ", ones_wrong)
    print("Avg. Time: ", avg_time, " seconds.")
    return correct_predictions, [
        len(test_data_combined), correct_predictions, wrong_predictions, zeros_predicted, ones_predicted,
        zeros_correct, ones_correct, zeros_wrong, ones_wrong, avg_time
    ]


# In[6]:


def experiment(training_set_distribuitions, tuple_sizes, bleaching_mode = [False]):
    output_file = "insights/results_experiment" + datetime.now().strftime('%Y%m%d%H%M%S') + ".csv"
    file = open(output_file, "w")
    file.write("data_distribution;tuple_size;bleaching_active;total_training_time;avg_in_sample_evaluation_time;total_in_sample_evaluation_time;avg_out_sample_evaluation_time;total_out_sample_evaluation_time;total_training_data;total_correct_training;" +
               "percent_correct_training;total_approved_training;correctly_approved_training;wrongly_approved_training;" +
               "percent_approved_correctly_training;total_reproved_training;correctly_reproved_training;" +
               "wrongly_reproved_training;percent_reproved_correctly_training;total_test_data;total_correct_test;" +
               "percent_correct_test;total_approved_test;correctly_approved_test;wrongly_approved_test;" +
               "percent_approved_correctly_test;total_reproved_test;correctly_reproved_test;" +
               "wrongly_reproved_test;percent_reproved_correctly_test;total_ties;ties_for_zeros;ties_for_ones\n"
              )
    file.close()
    
    
    for training_set_distribuition in training_set_distribuitions:
        print("\nTraining with a training set distribution of ", 
              training_set_distribuition[0], training_set_distribuition[1],
              " for approved and repproved, respectively.\n")
        training_input, expected_output, training_combined, test_input, test_expected_output, test_combined, training_set, test_set = getData(training_set_distribuition[0], training_set_distribuition[1])
        
        for a_tuple_size in tuple_sizes:
            print("Training with a tupple of size: ", a_tuple_size)
            
            for bleaching in bleaching_mode:
                print("Bleaching is set to: ", bleaching, "\n")

                training_time = time.time()
                wann = train(training_input, expected_output, a_tuple_size, bleaching)
                training_time = time.time() - training_time
                
                in_sample_evaluation_time = time.time()
                in_sample_performance, in_sample_additional_info =  evaluate_performance(wann, training_combined)
                in_sample_evaluation_time = time.time() - in_sample_evaluation_time
                
                # Evaluates Guilherme's wisard implementation
                print("In-sample performance: ", float(in_sample_performance) / float(len(training_combined)))
                print("Ones distribution: ", float(training_set["project_is_approved"].sum()) / float(len(training_set["project_is_approved"])))
                print("Ones: ", training_set["project_is_approved"].sum(), "Zeros: ", training_set["project_is_approved"].sum() - len(training_set["project_is_approved"]))
                print("\n")
                
                out_sample_evaluation_time = time.time()
                out_sample_performance, out_sample_additional_info =  evaluate_performance(wann, test_combined)
                out_sample_evaluation_time = time.time() - out_sample_evaluation_time
                
                print("Expected out-sample performance: ", float(out_sample_performance) / float(len(test_combined)))
                print("Ones distribution: ", float(test_set["project_is_approved"].sum()) / float(len(test_set["project_is_approved"])))
                print("Ones: ", test_set["project_is_approved"].sum(), "Zeros: ", (test_set["project_is_approved"].sum() - len(test_set["project_is_approved"])), "\n\n")
                
                line = ""
                line_contents = [
                    # training / test
                    str(training_set_distribuition[0]) + "/" + str(training_set_distribuition[1]) + ";",
                    # tuple size
                    str(a_tuple_size) + ";",
                    # bleaching active or not
                    str(bleaching) + ";",
                    # training time in seconds
                    str(training_time) + ";",
                    # in sample pattern average evaluation time in seconds
                    str(in_sample_additional_info[12]) + ";",
                    # in sample total evaluation time in seconds
                    str(in_sample_evaluation_time) + ";",
                    # out sample pattern average evaluation time in seconds
                    str(out_sample_additional_info[12]) + ";",
                    # out sample total evaluation time in seconds
                    str(out_sample_evaluation_time) + ";",
                    
                    # total traning observations
                    str(training_set_distribuition[0] + training_set_distribuition[1]) + ";",
                    # total of correct prediction in the training dataset
                    str(in_sample_performance) + ";",
                    # percentage of right answers
                    str(float(in_sample_performance) / float(len(training_combined))) + ";",
                    # total approved in the training dataset
                    str(training_set["project_is_approved"].sum()) + ";",
                    # total approved correctly predicted in the training dataset
                    str(in_sample_additional_info[6]) + ";",
                    # total approved wrongly predicted in the training dataset
                    str(in_sample_additional_info[8]) + ";",
                    # percentage of approved projects predicted correctly in the training dataset
                    str(float(in_sample_additional_info[6]) / float(training_set["project_is_approved"].sum())) + ";",
                    # total reproved in the training dataset
                    str((training_set["project_is_approved"].sum() - len(training_set["project_is_approved"])) * -1) + ";",
                    # total reproved correctly predicted in the training dataset
                    str(in_sample_additional_info[5]) + ";",
                    # total reproved wrongly predicted in the training dataset
                    str(in_sample_additional_info[7]) + ";",
                    # percentage of reproved projects predicted correctly in the training dataset
                    str(float(in_sample_additional_info[5]) / float((training_set["project_is_approved"].sum() - len(training_set["project_is_approved"])) * -1)) + ";",

                    
                    # total test observations
                    str(len(test_set["project_is_approved"])) + ";",
                    # total of correct prediction in the test dataset
                    str(out_sample_performance) + ";",
                    # percentage of right answers
                    str(float(out_sample_performance) / float(len(test_combined))) + ";",
                    # total approved in the test dataset
                    str(test_set["project_is_approved"].sum()) + ";",
                    # total approved correctly predicted in the test dataset
                    str(out_sample_additional_info[6]) + ";",
                    # total approved wrongly predicted in the test dataset
                    str(out_sample_additional_info[8]) + ";",
                    # percentage of approved projects predicted correctly in the test dataset
                    str(float(out_sample_additional_info[6]) / float(test_set["project_is_approved"].sum())) + ";",
                    # total reproved in the test dataset
                    str((test_set["project_is_approved"].sum() - len(test_set["project_is_approved"])) * -1) + ";",
                    # total reproved correctly predicted in the training dataset
                    str(out_sample_additional_info[5]) + ";",
                    # total reproved wrongly predicted in the training dataset
                    str(out_sample_additional_info[7]) + ";",
                    # percentage of reproved projects predicted correctly in the training dataset
                    str(float(out_sample_additional_info[5]) / float((test_set["project_is_approved"].sum() - len(test_set["project_is_approved"])) * -1)) + ";",
                    
                    # total ties
                    str(out_sample_additional_info[9]) + ";",
                    # total ties when prediction should have been zero
                    str(out_sample_additional_info[10]) + ";",
                    # total ties when prediction should have been one
                    str(out_sample_additional_info[11]) + ";",
                    
                    "\n",
                    #str() + ";",
                ]

                for content in line_contents:
                    #print(content)
                    line = line + content

                print(line)
                
                file = open(output_file, "a+")
                file.write(line)
                file.close()

        
    


# In[7]:


tuple_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 25, 30, 50, 100]
#tuple_sizes = [2]
training_set_distribuitions = [[5, 5], [10, 10], [20, 20], [30, 30], [50, 50], [75, 75], [86, 14], [100, 100]]
#training_set_distribuitions = [[100, 100]]

experiment(training_set_distribuitions, tuple_sizes, [False, True])

