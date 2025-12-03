from pathlib import Path
import pandas as pd

corpus_name = 'ENNI'


dfs = []
for group in [('SLI', 'A'), ('SLI', 'B'), ('TD', 'A'), ('TD', 'B')]:
    cha_files = sorted([str(p) for p in Path(corpus_name, group[0], group[1]).glob("*.cha")])
    subjects = [Path(f).name.replace('.cha', '') for f in cha_files]
    df_filenames = pd.DataFrame({        
        'group': group[0],
        'sub_group': group[1],
        'subject': subjects,
        'filename': cha_files
    }) #.to_csv(f'{corpus_name}_{group[0]}_{group[1]}.csv', index=False)
    # print(df_filenames)
    dfs.append(df_filenames)
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
stratify_cols = df_all[['group', 'sub_group']].astype(str).agg('-'.join, axis=1)

train_df, temp_df = train_test_split(
    df_all,
    test_size=(1 - train_ratio),
    stratify=stratify_cols,
    random_state=42
)

# For val and test split from temp
temp_stratify_cols = temp_df[['group', 'sub_group']].astype(str).agg('-'.join, axis=1)
val_test_ratio = val_ratio / (val_ratio + test_ratio)
val_df, test_df = train_test_split(
    temp_df,
    test_size=(1 - val_test_ratio),
    stratify=temp_stratify_cols,
    random_state=42
)
print("Train:", len(train_df), "Dev:", len(val_df), "Test:", len(test_df))

# train_df를 'group'과 'sub_group' 기준으로 소팅
train_df = train_df.sort_values(by=['group', 'sub_group'])
val_df = val_df.sort_values(by=['group', 'sub_group'])
test_df = test_df.sort_values(by=['group', 'sub_group'])

# Save results
train_df.to_csv(f"{corpus_name}_train.csv", index=False)
val_df.to_csv(f"{corpus_name}_dev.csv", index=False)
test_df.to_csv(f"{corpus_name}_test.csv", index=False)
