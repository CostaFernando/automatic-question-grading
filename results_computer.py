from scipy.stats import f_oneway, pearsonr
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def compute_std(data):
    std_dev_grades = data[['GPT-4 Grade', 'GPT-3.5 Grade', 'Llama 2 Chat 70B', 'Llama 2 Chat 13B']].std()
    return std_dev_grades

def compute_anova(data):
    anova_result = f_oneway(data['GPT-4 Grade'], data['GPT-3.5 Grade'], data['Llama 2 Chat 70B'], data['Llama 2 Chat 13B'])
    return anova_result

def compute_tukey(data):
    tukey_result = pairwise_tukeyhsd(endog=data['Embeddings similarity'], groups=data['Student Profile'], alpha=0.05)
    return tukey_result

def compute_mae_rmse(dataframe, target_column, comparison_column):
    mae = mean_absolute_error(dataframe[target_column], dataframe[comparison_column])
    rmse = np.sqrt(mean_squared_error(dataframe[target_column], dataframe[comparison_column]))
    return mae, rmse

def compute_embedding_grades_correlations(data, models):
    correlations = {}
    for model in models:
        corr, _ = pearsonr(data[model], data['Embeddings similarity'])
        correlations[model] = corr

    return correlations