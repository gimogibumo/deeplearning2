from pathlib import Path
import pandas as pd

corpus_name = 'gillam'


groups = ['SLI', 'TD']
sub_groups = ['5m', '5f', '6m','6f', '7m', '7m', '8m', '8f', '9m', '9f', '10m', '10f', '11m', '11f']


dfs = []
for group in groups:
    for sub_group in sub_groups:
        cha_files = sorted([str(p) for p in Path(corpus_name, group, sub_group).glob("*.cha")])
        subjects = [Path(f).name.replace('.cha', '') for f in cha_files]
        df_filenames = pd.DataFrame({        
            'group': group,
            'age': int(sub_group[:-1]),
            'gender': sub_group[-1],
            'subject': subjects,
            'filename': cha_files
        }) #.to_csv(f'{corpus_name}_{group}_{sub_group}.csv', index=False)
        # print(df_filenames)
        if len(df_filenames) > 0:
            dfs.append(df_filenames)
        else:
            print(f"No files found for {group} {sub_group}")
        # print(cha_files)

df_all = pd.concat(dfs)
df_all.to_csv(f'{corpus_name}_all.csv', index=False)
print(df_all)


# Split dataset
from sklearn.model_selection import train_test_split

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1


# Stratified split based on group and sub_group
stratify_cols = df_all[['group', 'age']].astype(str).agg('-'.join, axis=1)
# fix (too few samples for age 11 in SLI)
stratify_cols = stratify_cols.replace('SLI-11', 'SLI-10')

train_df, temp_df = train_test_split(
    df_all,
    test_size=(1 - train_ratio),
    stratify=stratify_cols,
    random_state=42
)

# For val and test split from temp
temp_stratify_cols = temp_df[['group', 'age']].astype(str).agg('-'.join, axis=1)
val_test_ratio = val_ratio / (val_ratio + test_ratio)
val_df, test_df = train_test_split(
    temp_df,
    test_size=(1 - val_test_ratio),
    stratify=temp_stratify_cols,
    random_state=42
)
print("Train:", len(train_df), "Dev:", len(val_df), "Test:", len(test_df))

# train_df를 'group'과 'sub_group' 기준으로 소팅
train_df = train_df.sort_values(by=['group', 'age', 'gender', 'subject'])
val_df = val_df.sort_values(by=['group', 'age', 'gender', 'subject'])
test_df = test_df.sort_values(by=['group', 'age', 'gender', 'subject'])

# Save results
train_df.to_csv(f"{corpus_name}_train.csv", index=False)
val_df.to_csv(f"{corpus_name}_dev.csv", index=False)
test_df.to_csv(f"{corpus_name}_test.csv", index=False)
