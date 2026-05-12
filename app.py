import pandas as pd
df = pd.read_csv('data/archive/asl_landmarks_final.csv')
print(df['label'].unique())
print(df['label'].value_counts())