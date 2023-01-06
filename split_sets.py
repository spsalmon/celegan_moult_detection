import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import csv

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)
        
whole_database_folder = "/mnt/external.data/TowbinLab/spsalmon/moult_database/whole_database/fluo/"

training_database_folder = "/mnt/external.data/TowbinLab/spsalmon/moult_database/fluo/training/"
validation_database_folder = "/mnt/external.data/TowbinLab/spsalmon/moult_database/fluo/validation/"


if not os.path.exists(training_database_folder):
    os.makedirs(training_database_folder)
if not os.path.exists(validation_database_folder):
    os.makedirs(validation_database_folder)

training_labels_csv = training_database_folder+"labels.csv"
validation_labels_csv = validation_database_folder+"labels.csv"

with open(training_labels_csv, 'a+', newline='') as csvfile:
    if os.stat(training_labels_csv).st_size == 0:
        writer = csv.writer(csvfile)
        writer.writerow(['img_id', 'label'])

with open(validation_labels_csv, 'a+', newline='') as csvfile:
    if os.stat(validation_labels_csv).st_size == 0:
        writer = csv.writer(csvfile)
        writer.writerow(['img_id', 'label'])

labels = pd.read_csv(whole_database_folder+"labels.csv")
indexes = np.arange(labels.shape[0])
# print(indexes)

train_indexes, validation_indexes = train_test_split(indexes, test_size=0.2)
print(len(train_indexes))
print(len(validation_indexes))

for index in train_indexes:
    shutil.copy(whole_database_folder+labels['img_id'].iloc[index], training_database_folder+labels['img_id'].iloc[index])
    append_list_as_row(training_labels_csv, labels.iloc[index].to_list())

for index in validation_indexes:
    shutil.copy(whole_database_folder+labels['img_id'].iloc[index], validation_database_folder+labels['img_id'].iloc[index])
    append_list_as_row(validation_labels_csv, labels.iloc[index].to_list())