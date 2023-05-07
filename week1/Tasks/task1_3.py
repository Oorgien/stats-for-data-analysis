#%%
import numpy as np
from scipy.stats import norm
# Давайте уточним правило трёх сигм.
# Утверждение: 99.7% вероятностной массы случайной величины X∼N(μ,σ^2) лежит в интервале μ±c⋅σ.
# Чему равно точное значение константы c?
# Округлите ответ до четырёх знаков после десятичной точки.
#%%
print ("3 sigma rule refined: %.4f" % np.round(norm.ppf(0.003/2), 4))

# В пятилетнем рандомизированном исследовании Гарвардской медицинской школы 11037 испытуемых через день
# принимали аспирин, а ещё 11034 — плацебо. Исследование было слепым, то есть, испытуемые не знали,
# что именно они принимают.
# За 5 лет инфаркт случился у 104 испытуемых, принимавших аспирин, и у 189 принимавших плацебо.
# Оцените, насколько вероятность инфаркта снижается при приёме аспирина. Округлите ответ до четырёх знаков
# после десятичной точки.
#%%

aspirin_n = 11037
placebo_n = 11034
aspirin_infarct_n = 104
placebo_infarct_n = 189

p_asp_inf = aspirin_infarct_n / aspirin_n
p_plac_inf = placebo_infarct_n / placebo_n

print("Probability diff: ", np.round(np.abs(p_asp_inf - p_plac_inf), 4))

#%%
# Постройте теперь 95% доверительный интервал для снижения вероятности
# инфаркта при приёме аспирина. Чему равна его верхняя граница?
# Округлите ответ до четырёх знаков после десятичной точки.


def proportions_confint_diff_ind(num1, total1, num2, total2, alpha=0.05):
    z = norm.ppf(1 - alpha / 2.)
    p1 = float(num1) / total1
    p2 = float(num2) / total2

    left_boundary = (p1 - p2) - z * np.sqrt(p1 * (1. - p1) / total1 + p2 * (1 - p2) / total2)
    right_boundary = (p1 - p2) + z * np.sqrt(p1 * (1. - p1) / total1 + p2 * (1 - p2) / total2)

    return left_boundary, right_boundary


from statsmodels.stats.proportion import proportion_confint
conf_interval_aspirin = proportion_confint(aspirin_infarct_n, aspirin_n, method="wilson")
conf_interval_placebo = proportion_confint(placebo_infarct_n, placebo_n, method="wilson")


print("95% доверительный интервал для снижения вероятности инфаркта при приёме аспирина:",
       np.round(proportions_confint_diff_ind(placebo_infarct_n, placebo_n, aspirin_infarct_n, aspirin_n), 4)
      )

#%%
# Продолжим анализировать данные эксперимента Гарвардской медицинской школы.
# Для бернуллиевских случайных величин X∼Ber(p) часто вычисляют величину p/(1−p), которая называется шансами
# (odds). Чтобы оценить шансы по выборке, вместо p нужно подставить p^. Например, шансы инфаркта в контрольной
# группе, принимавшей плацебо, можно оценить как
# (189/11034)/(1−189/11034)=189/(11034−189)≈0.0174
# Оцените, во сколько раз понижаются шансы инфаркта при регулярном приёме аспирина. Округлите ответ до четырёх
# знаков после десятичной точки.


def odds(num, total):
    return (num/total)/ (1 - num/total)


print(
    "Placebo to aspirin odds ratio:",
    np.round(odds(placebo_infarct_n, placebo_n) / odds(aspirin_infarct_n, aspirin_n), 4)
)

#%%
# Величина, которую вы оценили в предыдущем вопросе, называется отношением шансов.
# Постройте для отношения шансов 95% доверительный интервал с помощью бутстрепа.
# Чему равна его нижняя граница? Округлите ответ до 4 знаков после десятичной точки.
# Чтобы получить в точности такой же доверительный интервал, как у нас:
#   * составьте векторы исходов в контрольной и тестовой выборках так, чтобы в начале шли все единицы, а потом все нули;
#   * установите random seed=0;
#   * сделайте по 1000 псевдовыборок из каждой группы пациентов с помощью функции get_bootstrap_samples.


def get_bootstrap_samples(data, n_samples):
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples


def stat_intervals(stat, alpha=0.05):
    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])
    return boundaries


def odds(data):
    return (np.sum(data)/len(data))/ (1 - np.sum(data)/len(data))


np.random.seed(0)
aspirin = np.zeros(aspirin_n, )
aspirin[:aspirin_infarct_n] = 1

placebo = np.zeros(placebo_n, )
placebo[:placebo_infarct_n] = 1

aspirin_bootstrap = get_bootstrap_samples(aspirin, 1000)
placebo_bootstrap = get_bootstrap_samples(placebo, 1000)

aspirin_odds = list(map(odds, aspirin_bootstrap))
placebo_odds = list(map(odds, placebo_bootstrap))

odds = np.array(placebo_odds) / np.array(aspirin_odds)

print(np.round(stat_intervals(odds), 4))