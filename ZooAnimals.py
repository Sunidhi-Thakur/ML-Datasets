# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 17:22:48 2021

@author: Sunidhi Thakur
"""

import random
# read in the file
#Load the data using zoo.data file
file = open('zoo.data', 'r')
# create a list to include all the lines in the text file
#Create an Empty list
data = list()
for line in file:
    #Iterate over all the lines in the file and append each line in the list
    data.append(line)

file.close()
# remove /n from end of each line
data = [i.strip() for i in data]

print(data[0:5])
print(data[-5:])

split_data = [d.split(',') for d in data]
animal_names = []
# Iterate over each row in split_data
for row in split_data:
    name = row.pop(0)
    animal_names.append(name)

# define a function to convert legs attribute to 0 - 5
def convert_legs_value(data):
    value_list = ['0', '2', '4', '5', '6', '8']
    convert_list = ['0', '1', '2', '3', '4', '5']
    for row in data:
        row[12] = convert_list[value_list.index(row[12])]
    return data

split_data = convert_legs_value(split_data)
print(split_data[0:5])

def train_test_split(data, test_size):

    # Checks if the object is of the specified type.
    if isinstance(test_size, float):
        test_size = round(test_size * len(data))
    test_indices = random.sample(range(len(data)), k=test_size)
    train_indices = [i for i in range(len(data)) if i not in test_indices]
    
    # Create the empty list
    test_df = []
    # Iterate over each index of test_indices
    for index in test_indices:
        test_d = data[index]
        # Append the test data to test_df
        test_df.append(test_d)
    
    train_df = []
    for index in train_indices:
        train_d = data[index]
        train_df.append(train_d)
    
    # return the train_df and test_df
    return train_df, test_df

random.seed(0)
# Call the train_test_split() method for test_size=30
train_data, test_data = train_test_split(split_data, test_size = 30)
# find the length of the train_data
train_sample_num = len(train_data)
attr_num = len(train_data[0])-1

# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
    # Create an empty dictionary
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        # Check if the class_value is not in dictionary, then create an empty list for the class_value
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    # return the dictionary
    return separated

separated = separate_by_class(train_data)

for key, value in separated.items():
    print(key)
    print(value)
    
def calculate_class_py(label):
    #find the length of label
    count = len(label)
    #find the py by dividing count by train_sample_num
    py = count/train_sample_num
    return py

def calculate_pxy(label):
    pxy = []
    for i in range(attr_num):
        if i != 12:
            count0 = 0
            count1 = 0
            for row in label:
                if row[i] == '0':
                    count0 += 1
                else:
                    count1 += 1
            cal = ((count0 + 0.1)/(len(label) + 0.2), (count1 + 0.1)/(len(label) + 0.2))
            pxy.append(cal)
        else:
            count0 = 0
            count1 = 0
            count2 = 0
            count3 = 0
            count4 = 0
            count5 = 0
            for row in label:
                if row[i] == '0':
                    count0 += 1
                elif row[i] == '1':
                    count1 += 1
                elif row[i] == '2':
                    count2 += 1
                elif row[i] == '3':
                    count3 += 1
                elif row[i] == '4':
                    count4 += 1
                else:
                    count5 += 1
            cal = ((count0 + 0.1)/(len(label) + 0.6), (count1 + 0.1)/(len(label) + 0.6), (count2 + 0.1)/(len(label) + 0.6),
                   (count3 + 0.1)/(len(label) + 0.6), (count4 + 0.1)/(len(label) + 0.6), (count5 + 0.1)/(len(label) + 0.6))
            pxy.append(cal)
            
    return pxy

def summarize_by_class(separated):
    model = dict()
    for class_value, rows in separated.items():
        pxy = calculate_pxy(rows)
        py = calculate_class_py(rows)
        
        model[class_value] = [pxy, py]
    return model

model = summarize_by_class(separated)

for key, possibilities in model.items():
    print(key)
    print(possibilities)

def calculate_class_probabilities(model, sample):
    probabilities = dict()
    for class_value, class_summaries in model.items():
        probabilities[class_value] = class_summaries[1]
        for i in range(attr_num):
            probabilities[class_value] *= class_summaries[0][i][int(sample[i])]
    return probabilities

print(calculate_class_probabilities(model, test_data[0]))

# Predict the class for a given sample
def predict(model, sample):
    probabilities = calculate_class_probabilities(model, sample)
    best_label, best_prob = None, -1
    
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    
    return best_label

# get all the predictions
predictions = []
for row in test_data:
    best_label = predict(model, row)
    predictions.append(best_label)
# get all the real labels
actual = []
for row in test_data:
    real_label = row[-1]
    actual.append(real_label)
    
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

accuracy = accuracy_metric(actual, predictions)
print("The accuracy of predcition is: %.2f percent!" %accuracy )

































