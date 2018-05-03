
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
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
    
    #cleaned_df["category_1"] = cleaned_df["category_1"].fillna("Not Informed")
    cleaned_df["category_2"] = cleaned_df["category_2"].fillna("Not Informed")
    
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
    
    project_grade_category_mapping = {
        'Grades PreK-2':"00000000000000000000000000000000000000000000000000", 
        'Grades 3-5':"10000000000000000000000000000000000000000000000000", 
        'Grades 6-8':"11000000000000000000000000000000000000000000000000", 
        'Grades 9-12':"11100000000000000000000000000000000000000000000000"
    }

    teacher_prefix_mapping = {
        'Ms.':"11110000000000000000000000000000000000000000000000", 
        'Mrs.':"11111000000000000000000000000000000000000000000000", 
        'Mr.':"11111100000000000000000000000000000000000000000000", 
        'Teacher':"11111110000000000000000000000000000000000000000000", 
        'Dr.':"11111111000000000000000000000000000000000000000000", 
        'unknown':"11111111100000000000000000000000000000000000000000"
    }

    n_previous_projects_mapping = {
        '0-1':"11111111110000000000000000000000000000000000000000",
        '2-5':"11111111111000000000000000000000000000000000000000",
        '6-10':"11111111111100000000000000000000000000000000000000",
        '11-25':"11111111111110000000000000000000000000000000000000",
        '26-50':"11111111111111000000000000000000000000000000000000",
        '51+':"11111111111111100000000000000000000000000000000000"
    }

    total_price_category_mapping = {
        "0-100":"11111111111111110000000000000000000000000000000000",
        "101-250":"11111111111111111000000000000000000000000000000000",
        "251-500":"11111111111111111100000000000000000000000000000000",
        "501-1000":"11111111111111111110000000000000000000000000000000",
        ">1000":"11111111111111111111000000000000000000000000000000"
    }
    
    month_mapping = {
        "1":"11111111111111111111100000000000000000000000000000",
        "2":"11111111111111111111110000000000000000000000000000",
        "3":"11111111111111111111111000000000000000000000000000",
        "4":"11111111111111111111111100000000000000000000000000",
        "5":"11111111111111111111111110000000000000000000000000",
        "6":"11111111111111111111111111000000000000000000000000",
        "7":"11111111111111111111111111100000000000000000000000",
        "8":"11111111111111111111111111110000000000000000000000",
        "9":"11111111111111111111111111111000000000000000000000",
        "10":"11111111111111111111111111111100000000000000000000",
        "11":"11111111111111111111111111111110000000000000000000",
        "12":"11111111111111111111111111111111000000000000000000"
    }
    
    quarter_mapping = {
        "1":"11111111111111111111111111111111100000000000000000",
        "2":"11111111111111111111111111111111110000000000000000",
        "3":"11111111111111111111111111111111111000000000000000",
        "4":"11111111111111111111111111111111111111111111100000"
    }
    
    category_mapping = {
        "Not Informed":"11111111111111111111111111111111111100000000000000",
        "Applied Learning":"11111111111111111111111111111111111110000000000000",
        "Health & Sports":"11111111111111111111111111111111111111000000000000",
        "History & Civics":"11111111111111111111111111111111111111100000000000",
        "Literacy & Language":"11111111111111111111111111111111111111110000000000",
        "Math & Science":"11111111111111111111111111111111111111111000000000",
        "Music & The Arts":"11111111111111111111111111111111111111111100000000",
        "Special Needs":"11111111111111111111111111111111111111111110000000",
        "Warmth":"11111111111111111111111111111111111111111111000000",
        "Care & Hunger":"11111111111111111111111111111111111111111111000000", # Equals to warmth, because they are the same thing
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
        
        bits_string = ""
        bits_string = project_grade_category_mapping[row["project_grade_category"]]
        bits_string = bits_string + teacher_prefix_mapping[row["teacher_prefix"]]
        bits_string = bits_string + n_previous_projects_mapping[row["n_previous_projects"]]
        bits_string = bits_string + total_price_category_mapping[row["total_price_category"]]
        
        bits_string = bits_string + month_mapping[str(row["month"])]
        bits_string = bits_string + quarter_mapping[str(row["quarter"])]
        bits_string = bits_string + category_mapping[row["category_1"]]
        bits_string = bits_string + category_mapping[row["category_2"]]
        
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

    train = train_data.sample(n=10000,random_state=200)
    print("Distribution over a random sample of 10000 observations used to get the observations to train the classifier: ",
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
    wann = Wisard(tuple_size, 3546)
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
    for combined in test_data_combined:
        prediction = wann.predict(combined[0])
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
    
    print("Number of observations: ", len(test_data_combined))
    print("Predicted correctly: ", correct_predictions)
    print("Predicted wrongly: ", wrong_predictions)
    print("Predicted zeros: ", zeros_predicted)
    print("Predicted ones: ", ones_predicted)
    print("Zeros correct: ", zeros_correct)
    print("Ones correct: ", ones_correct)
    print("Zeros wrong: ", zeros_wrong)
    print("Ones Wrong: ", ones_wrong)
    return correct_predictions

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
    
    print("Number of observations: ", len(test_data_combined))
    print("Predicted correctly: ", correct_predictions)
    print("Predicted wrongly: ", wrong_predictions)
    print("Predicted zeros: ", zeros_predicted)
    print("Predicted ones: ", ones_predicted)
    print("Zeros correct: ", zeros_correct)
    print("Ones correct: ", ones_correct)
    print("Zeros wrong: ", zeros_wrong)
    print("Ones Wrong: ", ones_wrong)
    return correct_predictions


# In[6]:


def experiment(training_set_distribuitions, tuple_sizes, bleaching = False):
    for training_set_distribuition in training_set_distribuitions:
        print("\nTraining with a training set distribution of ", 
              training_set_distribuition[0], training_set_distribuition[1],
              " for approved and repproved, respectively.\n")
        training_input, expected_output, training_combined, test_input, test_expected_output, test_combined, training_set, test_set = getData(training_set_distribuition[0], training_set_distribuition[1])
        
        for a_tuple_size in tuple_sizes:
            print("Training with a tupple of size: ", a_tuple_size)
            print("Bleaching is set to: ", bleaching, "\n")
            
            wann = train(training_input, expected_output, a_tuple_size, bleaching)
            
            # Evaluates Guilherme's wisard implementation
            print("In-sample performance: ", float(evaluate_performance(wann, training_combined)) / float(len(training_combined)))
            print("Ones distribution: ", float(training_set["project_is_approved"].sum()) / float(len(training_set["project_is_approved"])))
            print("Ones: ", training_set["project_is_approved"].sum(), "Zeros: ", training_set["project_is_approved"].sum() - len(training_set["project_is_approved"]))
            print("\n")
            print("Expected out-sample performance: ", float(evaluate_performance(wann, test_combined)) / float(len(test_combined)))
            print("Ones distribution: ", float(test_set["project_is_approved"].sum()) / float(len(test_set["project_is_approved"])))
            print("Ones: ", test_set["project_is_approved"].sum(), "Zeros: ", (test_set["project_is_approved"].sum() - len(test_set["project_is_approved"])), "\n\n")
            
        
    


# In[7]:


tuple_sizes = [1, 2, 4, 5, 7, 10, 25, 50]
training_set_distribuitions = [[10, 10], [50, 50], [75, 75], [86, 14]]

experiment(training_set_distribuitions, tuple_sizes)

