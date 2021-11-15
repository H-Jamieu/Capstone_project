import pandas as pd
from sklearn.metrics import mean_squared_error
import csv
from tqdm import tqdm

def cal_diversity(group_data):
    index_list = group_data.index
    diversity = []
    for i in tqdm(range(len(index_list))):
        for j in range(i + 1, len(index_list)):
            set1 = group_data.loc[index_list[i]]
            set2 = group_data.loc[index_list[j]]
            intersect_num = len(set1.intersection(set2))
            diversity.append(1 - intersect_num / len(set1))
    diversity = sum(diversity) / (len(diversity))
    return diversity

''' 衡量模型拟合能力指标: 使用真实电影数据进行检验, 即userid-movieid组合是真实存在的 '''
def evaluate_fit_metrics(test_path, pred_data):
    # 输入预测结果数据要求包含4列，分别为userid，movieid，labels，preds
    data = pd.read_csv(test_path)
    data.rename({'rating': 'labels'}, axis=1, inplace=True)
    pred = pd.merge(data, pred_data,  how='left', left_on=['userId','movieId'], right_on = ['userId','movieId'])
    pred.fillna(3.5, inplace=True)
    # 总体均方误差和用户维度均方误差
    mse = mean_squared_error(y_true=pred['labels'], y_pred=pred['rating'])
    return {
        'mse': mse
    }

def get_statistic(single_user_generes_data):
    mean = single_user_generes_data.iloc[:,~single_user_generes_data.columns.isin(['userId'])].mean()
    median = single_user_generes_data.iloc[:,~single_user_generes_data.columns.isin(['userId'])].median()
    return mean[0], median[0]

''' 衡量模型推荐多样性: 将对目标用户计算每一个电影的评分，取TOP K个电影作为推荐对象，这些电影用户未必真实看过，所以没有真实的电影评分，无法用于衡量拟合能力 '''
def evaluate_diversity_metrics(preds_path, movie_path, test_path, K = 10):
    # 输入预测结果数据要求包含4列，分别为userid，movieid，labels，preds
    data = pd.read_csv(preds_path)
    mses = evaluate_fit_metrics(test_path,data)
    # 对每个用户获取预估评分最高的TOP K个结果
    data = data.sort_values(by=['rating'], ascending=False).groupby(by=['userId']).head(K)
    movie = pd.read_csv(movie_path, header=0)
    movie.columns = ['movieId', 'title', 'genres']

    # 拼接电影类别信息
    data = pd.merge(left=data, right=movie, how='left', left_on='movieId', right_on='movieId')
    data['genres'].fillna("", inplace=True)
    user_movie_data = data.groupby(by=['userId']).apply(lambda x: set(x['movieId']))
    # 电影多样性: 比较不同用户之间推荐Top K电影的多样性
    user_genres_data = data.groupby(by=['userId']).apply(lambda x: "|".join(x['genres']))
    single_user_genres_data = user_genres_data.apply(lambda x: len(set(x.split('|'))) / len(x.split('|')))
    ud_mean, ud_median = get_statistic(single_user_genres_data.to_frame())
    movie_diversity = cal_diversity(user_movie_data)

    # 电影类比多样性：比较单个用户推荐Top K电影类别的多样性--去重电影类别数量 / 电影类别数量

    # 电影类比多样性：比较不同用户之间推荐Top K电影类别的多样性
    user_genres_data = user_genres_data.apply(lambda x: set(x.split('|')))
    genres_diversity = cal_diversity(user_genres_data)
    return {
        'movie_diversity': movie_diversity,
        'user_diversity_mean': ud_mean,
        'user_diversity_median': ud_median,
        'genres_diversity':genres_diversity,
        'mse':mses['mse'],
    }

def save_output(result, flag_a,flag_b):
    a_file = open("Data/Metrics/"+flag_a+flag_b+"_5000.csv", "w")

    writer = csv.writer(a_file)
    for k,v in result.items():
        writer.writerow([k, v])

    a_file.close()



if __name__ == '__main__':
    flag_a = 'top'
    flag_b = ''
    PRED_PATH = 'Data/predictions/out_'+flag_a+flag_b+'_5000.csv'
    MOVIE_PATH = 'Data/movie_data/movies.csv'
    TEST_PATH = 'Data/user_data/test_syn_5000.csv'
    #fit_result = evaluate_fit_metrics(preds_path='Data/predictions/out_syn_5000.csv')
    #print(fit_result)
    diversity_result = evaluate_diversity_metrics(preds_path=PRED_PATH, movie_path=MOVIE_PATH, test_path=TEST_PATH)
    save_output(diversity_result, flag_a, flag_b)