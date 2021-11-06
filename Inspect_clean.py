import pandas as pd

def load_clean(flag = 'syn'):
    train_df = pd.read_csv('Data/User_data/train_'+flag+'_5000.csv')
    test_df = pd.read_csv('Data/User_data/test_'+flag+'_5000.csv')
    print(len(test_df))
    ur = test_df['userId'].value_counts()
    ur_cut = ur.index.to_list()[-80:]
    train_df = train_df[~train_df['userId'].isin(ur_cut)]
    test_df = test_df[~test_df['userId'].isin(ur_cut)]
    mov_train = train_df['movieId'].unique()
    mov_test = test_df['movieId'].unique()
    dpl = []
    for m in mov_test:
        if m not in mov_train:
            dpl.append(m)
    print(len(dpl))
    test_df = test_df[~test_df['movieId'].isin(dpl)]
    print(len(test_df))
    test_df.to_csv('Data/User_data/test_'+flag+'_5000.csv')
    train_df.to_csv('Data/User_data/train_'+flag+'_5000.csv')

load_clean('syn')

# Recommend ALL Movies' rating to user
# Delete UNWATCHED MOVIE