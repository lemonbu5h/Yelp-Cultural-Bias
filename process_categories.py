import pandas as pd
from collections import Counter


def persist_cate_cnt(cate_data):
    word_cnt = {}
    for cate_line in cate_data:
        cate_str_lst = [cate.strip() for cate in cate_line.split(',')]
        for cate_str in cate_str_lst:
            if word_cnt.get(cate_str, None) is None:
                word_cnt[cate_str] = 0
            else:
                word_cnt[cate_str] += 1
    most_common = Counter(word_cnt).most_common()
    with open('./output.txt', 'w') as file:
        for item in most_common:
            file.write(f'{item[0]}  {item[1]}\n')


def filter_cul_data(data):
    res_df = pd.DataFrame()
    match_lst = ['American (', 'Mexican', 'Chinese', 'Italian', 'Japanese', 'Thai', 'Vietnamese',
                 'Indian', 'Korean', 'Greek', 'Canadian', 'French', 'Pakistani', 'Spanish']
    for id, row in data.iterrows():
        for match_item in match_lst:
            if row['categories'].__contains__(match_item):
                cur_row = row.copy(deep=True)
                cur_row['categories'] = match_item
                res_df = res_df.append(cur_row)
    res_df.to_csv('selected_data_v2.csv', index=False)


file_path = '/Volumes/Ark/yelp_dataset/generated/all_features_2018_v2.csv'
data = pd.read_csv(file_path)

# persist_cate_cnt(data['categories'])

filter_cul_data(data)
