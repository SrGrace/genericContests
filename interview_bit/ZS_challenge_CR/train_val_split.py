import pandas as pd

file_path = '/home/srgrace/Downloads/Cristano_Ronaldo_Final_v1/data.csv'
train_csv = "/home/srgrace/Downloads/Cristano_Ronaldo_Final_v1/train.csv"
val_csv = "/home/srgrace/Downloads/Cristano_Ronaldo_Final_v1/val.csv"

df = pd.read_csv(file_path, index_col=[0])
df['shot_id_number'] = df['shot_id_number'].ffill() + \
                       df['shot_id_number'].groupby(df['shot_id_number'].notnull().cumsum()).cumcount()*1

df1 = pd.isna(df)
mask = df1['is_goal'] == False
df_train = df[mask]
df_val = df[~mask]

df_train.to_csv(train_csv, index=False)
df_val.to_csv(val_csv, index=False)



