#%%
# Уровень кальция в крови здоровых молодых женщин равен в среднем 9.5 милиграммам
# на децилитр и имеет характерное стандартное отклонение 0.4 мг/дл.
# В сельской больнице Гватемалы для 160 здоровых беременных женщин при первом обращении
# для ведения беременности был измерен уровень кальция; среднее значение составило 9.57 мг/дл.
# Можно ли утверждать, что средний уровень кальция в этой популяции отличается от 9.5?

# Посчитайте достигаемый уровень значимости. Поскольку известны только среднее и дисперсия, а не сама выборка,
# нельзя использовать стандартные функции критериев —
# нужно реализовать формулу достигаемого уровня значимости самостоятельно.

# Округлите ответ до четырёх знаков после десятичной точки.

from scipy import stats
import numpy as np
from scipy.stats import norm

mu_exp = 9.5
sigma = 0.4
n = 160

mu = 9.57

statistic = (mu - mu_exp) / (sigma /np.sqrt(n))
p = 2.*(1.-norm.cdf(statistic))

print ("P value:", np.round(p, 4))

#%%
# Имеются данные о стоимости и размерах 53940 бриллиантов.
# Отделите 25% случайных наблюдений в тестовую выборку с помощью функции sklearn.model_selection.train_test_split
# (зафиксируйте random state = 1). На обучающей выборке настройте две регрессионные модели:
#
#   1) линейную регрессию с помощью LinearRegression без параметров
#   2) случайный лес из 10 деревьев с помощью RandomForestRegressor с random_state=1.
#
#   Какая из моделей лучше предсказывает цену бриллиантов?
#   Сделайте предсказания на тестовой выборке, посчитайте модули отклонений предсказаний от истинных цен.
#   Проверьте гипотезу об одинаковом среднем качестве предсказаний, вычислите достигаемый уровень значимости.
#   Отвергается ли гипотеза об одинаковом качестве моделей против двусторонней альтернативы на уровне значимости α=0.05?

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
from statsmodels.stats.weightstats import *
frame = pd.read_csv("week2/Tasks/diamonds.txt", sep="\t", header=0)

y = frame['price']
del frame['price']

X_train, X_test, y_train, y_test = train_test_split(frame, y, test_size=0.25, random_state=1)
model1 = LinearRegression()
model2 = RandomForestRegressor(random_state=1, n_estimators=10)

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

pred1 = np.abs(model1.predict(X_test) - y_test)
pred2 = np.abs(model2.predict(X_test) - y_test)

stats.probplot(pred1 - pred2, dist = "norm", plot = plt)
plt.show()

print ("Shapiro-Wilk normality test, W-statistic: %f, p-value: %f" % stats.shapiro(pred1-pred2))
stats.ttest_rel(pred1, pred2)

#%%
# В предыдущей задаче посчитайте 95% доверительный интервал для разности средних абсолютных
# ошибок предсказаний регрессии и случайного леса. Чему равна его ближайшая к нулю граница?
# Округлите до десятков (поскольку случайный лес может давать немного разные предсказания
# в зависимости от версий библиотек, мы просим вас так сильно округлить,
# чтобы полученное значение наверняка совпало с нашим).

print ("95%% confidence interval: [%f, %f]" % DescrStatsW(pred1-pred2).tconfint_mean())