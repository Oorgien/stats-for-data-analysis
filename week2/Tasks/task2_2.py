#%%

# В одном из выпусков программы "Разрушители легенд" проверялось, действительно ли заразительна зевота.
# В эксперименте участвовало 50 испытуемых, проходивших собеседование на программу.
# Каждый из них разговаривал с рекрутером; в конце 34 из 50 бесед рекрутер зевал. Затем испытуемых просили
# подождать решения рекрутера в соседней пустой комнате.
#
# Во время ожидания 10 из 34 испытуемых экспериментальной группы и 4 из 16 испытуемых контрольной начали зевать.
# Таким образом, разница в доле зевающих людей в этих двух группах составила примерно 4.4%. Ведущие заключили,
# что миф о заразительности зевоты подтверждён.
#
# Можно ли утверждать, что доли зевающих в контрольной и экспериментальной группах отличаются статистически значимо?
# Посчитайте достигаемый уровень значимости при альтернативе заразительности зевоты, округлите до четырёх знаков
# после десятичной точки.

import numpy as np

n = 50
x_yawned = 10
x = 34
y_yawned = 4
y = 16


def proportions_diff_z_stat_ind(x_sample, x, y_sample, y):
    p1 = float(x_sample) / x
    p2 = float(y_sample) / y
    P = float(p1 * x + p2 * y) / (x + y)

    return (p1 - p2) / np.sqrt(P * (1 - P) * (1. / x + 1. / y))


def proportions_diff_z_test(z_stat, alternative='two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")

    if alternative == 'two-sided':
        return 2 * (1 - scipy.stats.norm.cdf(np.abs(z_stat)))

    if alternative == 'less':
        return scipy.stats.norm.cdf(z_stat)

    if alternative == 'greater':
        return 1 - scipy.stats.norm.cdf(z_stat)


print("P value: ", np.round(proportions_diff_z_test(proportions_diff_z_stat_ind(x_yawned, x, y_yawned, y), "greater"), 4))

#%%

# Имеются данные измерений двухсот швейцарских тысячефранковых банкнот, бывших в обращении в первой половине XX века.
# Сто из банкнот были настоящими, и сто — поддельными. На рисунке ниже показаны измеренные признаки:

# Отделите 50 случайных наблюдений в тестовую выборку с помощью функции sklearn.cross_validation.train_test_split
# (зафиксируйте random state = 1). На оставшихся 150 настройте два классификатора поддельности банкнот:
#
# 1) логистическая регрессия по признакам x1, x2, x3
# 2) логистическая регрессия по признакам x4, x5, x6

# Каждым из классификаторов сделайте предсказания меток классов на тестовой выборке.
# Одинаковы ли доли ошибочных предсказаний двух классификаторов?
# Проверьте гипотезу, вычислите достигаемый уровень значимости.
# Введите номер первой значащей цифры (например, если вы получили
# 5.5 x 10^-8, нужно ввести 8).

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import scipy

banknotes = pd.read_csv("week2/Tasks/banknotes.txt", sep="\t", header=0)
y = banknotes['real']
del banknotes['real']

X_train, X_test, y_train, y_test = train_test_split(banknotes, y, test_size=50, random_state=1)

model1 = LogisticRegression()
model1.fit(X_train[["X1", "X2", "X3"]], y_train)
predictions1 = model1.predict(X_test[["X1", "X2", "X3"]])
wrong1 = 1 - accuracy_score(y_test, predictions1)

model2 = LogisticRegression()
model2.fit(X_train[["X4", "X5", "X6"]], y_train)
predictions2 = model2.predict(X_test[["X4", "X5", "X6"]])
wrong2 = 1 - accuracy_score(y_test, predictions2)


def proportions_diff_z_stat_rel(sample1, sample2):
    sample = list(zip(sample1, sample2))
    n = len(sample)

    f = sum([1 if (x[0] == 1 and x[1] == 0) else 0 for x in sample])
    g = sum([1 if (x[0] == 0 and x[1] == 1) else 0 for x in sample])

    return float(f - g) / np.sqrt(f + g - float((f - g) ** 2) / n)

pred_is_true1 = list(map(lambda x: 1 if x[0] == x[1] else 0, zip(predictions1, y_test)))
pred_is_true2 = list(map(lambda x: 1 if x[0] == x[1] else 0, zip(predictions2, y_test)))

print("P-value: ",
      proportions_diff_z_test(
          proportions_diff_z_stat_rel(
              pred_is_true1,
              pred_is_true2
          ),
      ))

#%%
import pandas as pd

frame = pd.read_csv("week2/Tasks/banknotes.txt", sep="\t", header=0)
frame.head()

#

from sklearn.model_selection import train_test_split

y = frame["real"]
x = frame.drop("real", axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=50, random_state=1)

# На оставшихся 150 настройте два классификатора поддельности банкнот:
#  1. логистическая регрессия по признакам X1,X2,X3;
#  2. логистическая регрессия по признакам X4,X5,X6.

from sklearn.linear_model import LogisticRegression

x_train_1_3 = x_train[["X1", "X2", "X3"]]
x_test_1_3 = x_test[["X1", "X2", "X3"]]

x_train_4_6 = x_train[["X4", "X5", "X6"]]
x_test_4_6 = x_test[["X4", "X5", "X6"]]

regression_1_3 = LogisticRegression().fit(x_train_1_3, y_train)
regression_4_6 = LogisticRegression().fit(x_train_4_6, y_train)

predictions_1_3 = regression_1_3.predict(x_test_1_3)
predictions_4_6 = regression_4_6.predict(x_test_4_6)

predictions_res_1_3 = list(map(lambda x: 1 if x[0] == x[1] else 0, zip(predictions_1_3, y_test)))
predictions_res_4_6 = list(map(lambda x: 1 if x[0] == x[1] else 0, zip(predictions_4_6, y_test)))


def proportions_diff_confint_rel(sample1, sample2, alpha=0.05):
    z = scipy.stats.norm.ppf(1 - alpha / 2.)
    sample = list(zip(sample1, sample2))
    n = len(sample)

    f = sum([1 if (x[0] == 1 and x[1] == 0) else 0 for x in sample])
    g = sum([1 if (x[0] == 0 and x[1] == 1) else 0 for x in sample])

    left_boundary = float(f - g) / n - z * np.sqrt(float((f + g)) / n ** 2 - float((f - g) ** 2) / n ** 3)
    right_boundary = float(f - g) / n + z * np.sqrt(float((f + g)) / n ** 2 - float((f - g) ** 2) / n ** 3)
    return (left_boundary, right_boundary)


def proportions_diff_z_stat_rel(sample1, sample2):
    sample = list(zip(sample1, sample2))
    n = len(sample)

    f = sum([1 if (x[0] == 1 and x[1] == 0) else 0 for x in sample])
    g = sum([1 if (x[0] == 0 and x[1] == 1) else 0 for x in sample])

    return float(f - g) / np.sqrt(f + g - float((f - g) ** 2) / n)


pvalue = proportions_diff_z_test(proportions_diff_z_stat_rel(predictions_res_1_3, predictions_res_4_6))
print("p-value: %f" % proportions_diff_z_test(proportions_diff_z_stat_rel(predictions_res_1_3, predictions_res_4_6)))

cinfidence_int_rel = proportions_diff_confint_rel(predictions_res_1_3, predictions_res_4_6)
print ("95%% confidence interval for a difference between predictions: [%.4f, %.4f]" \
      % (np.round(cinfidence_int_rel[0],4), np.round(cinfidence_int_rel[1], 4)))

#%%

# В предыдущей задаче посчитайте 95% доверительный интервал для разности долей ошибок двух классификаторов.
# Чему равна его ближайшая к нулю граница? Округлите до четырёх знаков после десятичной точки.

def proportions_diff_confint_rel(sample1, sample2, alpha=0.05):
    z = scipy.stats.norm.ppf(1 - alpha / 2.)
    sample = list(zip(sample1, sample2))
    n = len(sample)

    f = sum([1 if (x[0] == 1 and x[1] == 0) else 0 for x in sample])
    g = sum([1 if (x[0] == 0 and x[1] == 1) else 0 for x in sample])

    left_boundary = float(f - g) / n - z * np.sqrt(float((f + g)) / n ** 2 - float((f - g) ** 2) / n ** 3)
    right_boundary = float(f - g) / n + z * np.sqrt(float((f + g)) / n ** 2 - float((f - g) ** 2) / n ** 3)
    return (left_boundary, right_boundary)

print("95% confidence interval is: ", np.round(proportions_diff_confint_rel(pred_is_true1, pred_is_true2), 4))

#%%

# Ежегодно более 200000 людей по всему миру сдают стандартизированный экзамен GMAT при поступлении на программы MBA.
# Средний результат составляет 525 баллов, стандартное отклонение — 100 баллов.
#
# Сто студентов закончили специальные подготовительные курсы и сдали экзамен.
# Средний полученный ими балл — 541.4. Проверьте гипотезу о неэффективности программы против односторонней
# альтернативы о том, что программа работает. Отвергается ли на уровне значимости 0.05 нулевая гипотеза?
# Введите достигаемый уровень значимости, округлённый до 4 знаков после десятичной точки.

n = 200000.
mu = 525
std_s = 100
n_s = 100
mu_s = 541.4

def z_test(value, mu, std, n):
    return proportions_diff_z_test((value - mu)/(std/np.sqrt(n)), "greater")

print("P value: ", np.round(z_test(mu_s, mu, std_s, n_s), 4))

#%%
# Оцените теперь эффективность подготовительных курсов, средний балл 100 выпускников которых равен 541.5.
# Отвергается ли на уровне значимости 0.05 та же самая нулевая гипотеза против той же самой альтернативы?
# Введите достигаемый уровень значимости, округлённый до 4 знаков после десятичной точки.

n_s = 100
mu_s = 541.5

print("P value: ", np.round(z_test(mu_s, mu, std_s, n_s), 4))