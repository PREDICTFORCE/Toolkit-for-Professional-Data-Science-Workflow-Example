import os.path
from pathlib import Path

import pandas as pd

import argparse
import seaborn as sns
from matplotlib import pyplot as plt

# Read input params
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-train-data-path', type=str, help='an integer for the accumulator')
parser.add_argument('--input-test-data-path', type=str, help='an integer for the accumulator')
parser.add_argument('--output-plots-path', type=str, help='an integer for the accumulator')
args = parser.parse_args()
Path(args.output_plots_path).mkdir(parents=True, exist_ok=True)

train = pd.read_csv(args.input_train_data_path)

color = sns.color_palette('viridis')
pclass_survived = train.groupby('Pclass')['Survived'].count()
plt.subplots(figsize=(10, 5))
pclass_survived.plot(kind='bar', fontsize=12, color=color)
plt.xticks(rotation=0)
plt.xlabel('Pclass', fontsize=12)
plt.ylabel('Number of Passengers', fontsize=12)
plt.title('Distribution of passengers in Pclass', fontsize=12)
plt.savefig(os.path.join(args.output_plots_path, 'passengers_in_Pclass.png'))

f, ax = plt.subplots(figsize=(11, 9))
sns.countplot(x='Survived', hue='Pclass', data=train, palette='viridis').set(title='Survival Distribution by Pclass')
plt.savefig(os.path.join(args.output_plots_path, 'survival_distribution_by_Pclass.png'))

f, ax = plt.subplots(figsize=(11, 9))
g = sns.catplot(x="Pclass", hue="Sex", col="Survived",
                data=train, palette='viridis', kind="count")
plt.savefig(os.path.join(args.output_plots_path, 'survival_distribution_by_Pclass_and_gender.png'))

f, ax = plt.subplots(figsize=(11, 9))
corrDf = train.corr()
corr_plt = sns.heatmap(corrDf,
                       xticklabels=corrDf.columns,
                       yticklabels=corrDf.columns, annot=True, cmap='viridis')
plt.savefig(os.path.join(args.output_plots_path, 'correlation_matrix.png'))
