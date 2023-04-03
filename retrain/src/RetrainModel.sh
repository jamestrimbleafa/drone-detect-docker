#!/bin/bash
TRAIN_PATH=/app/DroneTrainDataset
TEST_PATH=/app/DroneTestDataset

# Check if that the datasets exist in /app/DroneTrainDataset and /app/DroneTestDataset (i.e. folders are not empty)
if [ -d "$TRAIN_PATH" ]; then
    if [ "$(ls -A $TRAIN_PATH)"]; then
        echo "Training Data found."
    else
        echo "Training Data not found."
        exit
    fi
else
    echo "Training Data not found."
    exit
fi

if [ -d "$TEST_PATH" ]; then
    if [ "$(ls -A $TEST_PATH)"]; then
        echo "Testing Data found."
    else
        echo "Testing Data not found."
        exit
    fi
else
    echo "Testing Data not found."
    exit
fi

# Check that the csv files exist.  If not, regenerate them.
if [ -f "$TRAIN_PATH/drone_train_labels.csv"]; then
    echo "Training CSV found."
else
    # If no CSV is found, rebuild the CSV file.
    echo "Training CSV not found.  Building CSV."
    python xml_to_csv.py $TRAIN_PATH/Drone_TrainSet_XMLs/ $TRAIN_PATH/drone_train_labels.csv
fi

if [ -f "$TEST_PATH/drone_test_labels.csv"]; then
    echo "Testing CSV found."
else
    # If no CSV is found, rebuild the CSV file.
    echo "Testing CSV not found.  Building CSV."
    python xml_to_csv.py $TEST_PATH/Drone_TestSet_XMLs/ $TEST_PATH/drone_test_labels.csv
fi


# Build the TFRecords from the csv files and images

# 
