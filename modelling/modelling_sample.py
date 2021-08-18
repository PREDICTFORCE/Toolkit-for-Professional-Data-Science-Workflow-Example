import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-train-data-path', type=str, help='an integer for the accumulator')
parser.add_argument('--input-test-data-path', type=str, help='an integer for the accumulator')
parser.add_argument('--output-predictions-data-path', type=str, help='an integer for the accumulator')
parser.add_argument('--output-model-path', type=str, help='an integer for the accumulator')
args = parser.parse_args()
Path(args.output_predictions_data_path).parent.mkdir(parents=True, exist_ok=True)
Path(args.output_model_path).parent.mkdir(parents=True, exist_ok=True)
Path(args.output_predictions_data_path).parent.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(args.input_train_data_path)
test = pd.read_csv(args.input_test_data_path)

train, valid = train_test_split(train, train_size=0.8, random_state=150)

x_train = train.drop(['Survived'], axis=1).copy()
y_train = train['Survived'].copy()
x_valid = valid.drop(['Survived'], axis=1).copy()
y_valid = valid['Survived'].copy()
x_test = test.drop(['Survived', 'PassengerId'], axis=1).copy()
y_test = test['Survived'].copy()

model = LogisticRegression(solver='liblinear', random_state=42)

model.fit(x_train, y_train)

pred_valid = model.predict(x_valid)

accuracy_score(y_valid, pred_valid)

confusion_matrix(y_valid, pred_valid)

print(classification_report(y_valid, pred_valid))

# Actual Test Prediction
pred_test = model.predict(x_test).astype(int)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': pred_test})
output.to_csv('predicted_output.csv', index=False)
print("Your submission was successfully saved!")

pickle.dump(model, open(args.output_model_path, 'wb'))
