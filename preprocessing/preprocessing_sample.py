import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer


def clean_data(data, features_to_clean):
    for feature in features_to_clean:
        data.drop(feature, axis=1, inplace=True)


def fulfill_missing_values(data, metadata=None):
    if metadata:
        age_median = metadata['age_median']
        fare_median = metadata['fare_median']
        embarked_value = metadata['embarked_value']
    else:
        age_median = data['Age'].median()
        fare_median = data['Fare'].median()
        embarked_value = data['Embarked'].mode()[0]
    data['Age'].fillna(age_median, inplace=True)
    data['Fare'].fillna(fare_median, inplace=True)
    data['Embarked'].fillna(embarked_value, inplace=True)
    return {
        'age_median': age_median,
        'fare_median': fare_median,
        'embarked_value': embarked_value
    }


def one_hot_encoding(data, features, metadata={}):
    assert len(features) > 0
    for feature in features:
        label_binarizer = LabelBinarizer()
        if feature in metadata:
            label_binarizer.classes_ = np.array(metadata[feature])
            labeled_features = label_binarizer.transform(data[feature])
        else:
            labeled_features = label_binarizer.fit_transform(data[feature])
        column_names_for_labeled_features = ['{}_{}'.format(feature, cls) for cls in label_binarizer.classes_] if len(
            label_binarizer.classes_) >= 3 else ['{}_{}'.format(feature, label_binarizer.classes_[0])]
        data = data.join(pd.DataFrame(labeled_features,
                                      columns=column_names_for_labeled_features,
                                      index=data.index))
        data.drop(feature, axis=1, inplace=True)
        metadata[feature] = label_binarizer.classes_.tolist()
    return data, metadata


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-train-data-path', type=str, help='an integer for the accumulator')
parser.add_argument('--input-test-data-path', type=str, help='an integer for the accumulator')
parser.add_argument('--output-train-data-path', type=str, help='an integer for the accumulator')
parser.add_argument('--output-test-data-path', type=str, help='an integer for the accumulator')
args = parser.parse_args()
Path(args.output_train_data_path).parent.mkdir(parents=True, exist_ok=True)
Path(args.output_test_data_path).parent.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(args.input_train_data_path)
test = pd.read_csv(args.input_test_data_path)

clean_data(data=train,
           features_to_clean=['PassengerId', 'Cabin', 'Name', 'Ticket'])
clean_data(data=test,
           features_to_clean=['Cabin', 'Name', 'Ticket'])

missing_values_fulfillment_metadata = fulfill_missing_values(data=train)
fulfill_missing_values(data=test,
                       metadata=missing_values_fulfillment_metadata)

train, encoding_metadata = one_hot_encoding(train,
                                            features=['Embarked', 'Sex'])
test, _ = one_hot_encoding(test,
                           features=['Embarked', 'Sex'],
                           metadata=encoding_metadata)

train.to_csv(args.output_train_data_path)
test.to_csv(args.output_test_data_path)
