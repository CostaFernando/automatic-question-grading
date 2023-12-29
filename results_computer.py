from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def compute_std(data):
    std_dev_grades = data[['GPT-4 Grade', 'GPT-3.5 Grade', 'Llama 2 Chat 70B', 'Llama 2 Chat 13B']].std()
    return std_dev_grades

def compute_anova(data):
    anova_result = f_oneway(data['GPT-4 Grade'], data['GPT-3.5 Grade'], data['Llama 2 Chat 70B'], data['Llama 2 Chat 13B'])
    return anova_result

def compute_tukey(data):
    tukey_result = pairwise_tukeyhsd(endog=data['Embeddings similarity'], groups=data['Student Profile'], alpha=0.05)
    return tukey_result

# Mapeando o perfil do aluno para as notas verdadeiras
data['True Grade'] = data['Student Profile'].map({'Insuficiente': 1, 'Básico': 2, 'Proficiente': 3})

# Calculando MAE e RMSE para cada modelo
models = ['GPT-4 Grade', 'GPT-3.5 Grade', 'Llama 2 Chat 70B', 'Llama 2 Chat 13B']
mae_rmse_results = {}

for model in models:
    mae = mean_absolute_error(data['True Grade'], data[model])
    rmse = np.sqrt(mean_squared_error(data['True Grade'], data[model]))
    mae_rmse_results[model] = {'MAE': mae, 'RMSE': rmse}

data['RMSE GPT-4 Grade'] = (data['GPT-4 Grade'] - data['True Grade']) ** 2
data['RMSE GPT-3.5 Grade'] = (data['GPT-3.5 Grade'] - data['True Grade']) ** 2
data['RMSE Llama 2 Chat 70B'] = (data['Llama 2 Chat 70B'] - data['True Grade']) ** 2
data['RMSE Llama 2 Chat 13B'] = (data['Llama 2 Chat 13B'] - data['True Grade']) ** 2

# Realizando ANOVA para os RMSEs
anova_result_rmse = stats.f_oneway(data['RMSE GPT-4 Grade'],
                                   data['RMSE GPT-3.5 Grade'],
                                   data['RMSE Llama 2 Chat 70B'],
                                   data['RMSE Llama 2 Chat 13B'])