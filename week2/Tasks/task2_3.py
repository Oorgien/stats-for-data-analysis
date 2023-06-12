#%%

# Давайте вернёмся к данным выживаемости пациентов с лейкоцитарной лимфомой из видео про критерий знаков:
#
# 49,58,75,110,112,132,151,276,281,362*
#
# Измерено остаточное время жизни с момента начала наблюдения (в неделях); звёздочка обозначает цензурирование сверху —
# исследование длилось 7 лет, и остаточное время жизни одного пациента, который дожил до конца наблюдения, неизвестно.
#
# Поскольку цензурировано только одно наблюдение, для проверки гипотезы H_0: med X = 200
# на этих данных можно использовать критерий знаковых рангов — можно считать, что время дожития последнего пациента
# в точности равно 362, на ранг этого наблюдения это никак не повлияет.
#
# Критерием знаковых рангов проверьте эту гипотезу против двусторонней альтернативы, введите достигаемый
# уровень значимости, округлённый до четырёх знаков после десятичной точки.
#
# 1) Для классификатора используйте solver='liblinear'
#
# 2) Укажите для критерия знаковых рангов Вилкоксона mode='approx'

import numpy as np
from scipy import stats
from statsmodels.stats.weightstats import zconfint

life_times = np.array([49,58,75,110,112,132,151,276,281,362]) #∗

medX = 200
print("Wilcoxon criterion pvalue result: %.4f" % np.round(stats.wilcoxon(life_times-medX, mode="approx").pvalue, 4))
# 0.2845

#%%

# В ходе исследования влияния лесозаготовки на биоразнообразие лесов острова Борнео собраны данные о количестве видов
# деревьев в 12 лесах, где вырубка не ведётся:
#
# 22,22,15,13,19,19,18,20,21,13,13,15,
#
# и в 9 лесах, где идёт вырубка:
#
# 17,18,18,15,12,4,14,15,10.
#
# Проверьте гипотезу о равенстве среднего количества видов в двух
# типах лесов против односторонней альтернативы о снижении биоразнообразия в вырубаемых лесах.
# Используйте ранговый критерий. Чему равен достигаемый уровень значимости?
# Округлите до четырёх знаков после десятичной точки.
#

no_cut_kinds = np.array([22,22,15,13,19,19,18,20,21,13,13,15])
cut_kinds = np.array([17,18,18,15,12,4,14,15,10])

print("Wilcoxon criterion pvalue result:", stats.mannwhitneyu(no_cut_kinds, cut_kinds, alternative="greater"))
# 0.0290

#%%
# 28 января 1986 года космический шаттл "Челленджер" взорвался при взлёте.
# Семь астронавтов, находившихся на борту, погибли. В ходе расследования причин катастрофы основной версией была
# неполадка с резиновыми уплотнительными кольцами в соединении с ракетными ускорителями.
# Для 23 предшествовавших rатастрофе полётов "Челленджера" известны температура воздуха и появление повреждений
# хотя бы у одного из уплотнительных колец.
# С помощью бутстрепа постройте 95% доверительный интервал для разности средних температур воздуха при запусках,
# когда уплотнительные кольца повреждались, и запусках, когда повреждений не было.
# Чему равна его ближайшая к нулю граница? Округлите до четырёх знаков после запятой.
#
# Чтобы получить в точности такой же доверительный интервал, как у нас:
#
# 1) установите random seed = 0 перед первым вызовом функции get_bootstrap_samples, один раз
#
# 2) сделайте по 1000 псевдовыборок из каждой выборки.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import scipy

challenger = pd.read_csv("week2/Tasks/challenger.txt", sep="\t", header=0)
temprature_incident = challenger[challenger["Incident"] == 1]["Temperature"].to_numpy()
temprature_ok = challenger[challenger["Incident"] == 0]["Temperature"].to_numpy()


def get_bootstrap_samples(data, n_samples):
    data_length = len(data)
    indexes = np.random.randint(0, data_length, (n_samples, data_length))
    return data[indexes]

def stat_intervals(stat, alpha):
    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])
    return boundaries

bootstrap_temprature_incident = get_bootstrap_samples(temprature_incident, 1000)
bootstrap_temprature_ok = get_bootstrap_samples(temprature_ok, 1000)

from statsmodels.stats.weightstats import zconfint, _zconfint_generic
np.random.seed(0)
print("96% conf interval: ",
      np.round(stat_intervals(np.mean(bootstrap_temprature_ok, axis=1)- np.mean(bootstrap_temprature_incident, axis=1), 0.05), 4))

# 1.4504

#%%

# На данных предыдущей задачи проверьте гипотезу об одинаковой средней температуре воздуха в дни,
# когда уплотнительный кольца повреждались, и дни, когда повреждений не было.
# Используйте перестановочный критерий и двустороннюю альтернативу.
# Чему равен достигаемый уровень значимости? Округлите до четырёх знаков после десятичной точки.
#
# Чтобы получить такое же значение, как мы:
#
# установите random seed = 0;
#

# возьмите 10000 перестановок.
import itertools


def permutation_t_stat_ind(sample1, sample2):
    return np.mean(sample1) - np.mean(sample2)


def get_random_combinations(n1, n2, max_combinations):
    index = list(range(n1 + n2))
    indices = set([tuple(index)])
    for i in range(max_combinations - 1):
        np.random.shuffle(index)
        indices.add(tuple(index))
    return [(index[:n1], index[n1:]) for index in indices]


def permutation_zero_dist_ind(sample1, sample2, max_combinations=None):
    joined_sample = np.hstack((sample1, sample2))
    n1 = len(sample1)
    n = len(joined_sample)

    if max_combinations:
        indices = get_random_combinations(n1, len(sample2), max_combinations)
    else:
        indices = [(list(index), filter(lambda i: i not in index, range(n))) \
                   for index in itertools.combinations(range(n), n1)]

    distr = [joined_sample[list(i[0])].mean() - joined_sample[list(i[1])].mean() \
             for i in indices]
    return distr


def permutation_test(sample, mean, max_permutations=None, alternative='two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")

    t_stat = permutation_t_stat_ind(sample, mean)

    zero_distr = permutation_zero_dist_ind(sample, mean, max_permutations)

    if alternative == 'two-sided':
        return sum([1. if abs(x) >= abs(t_stat) else 0. for x in zero_distr]) / len(zero_distr)

    if alternative == 'less':
        return sum([1. if x <= t_stat else 0. for x in zero_distr]) / len(zero_distr)

    if alternative == 'greater':
        return sum([1. if x >= t_stat else 0. for x in zero_distr]) / len(zero_distr)

np.random.seed(0)
print ("P value: ", permutation_test(temprature_ok, temprature_incident, max_permutations=10000))
# 0.007

