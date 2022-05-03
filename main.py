import numpy as np
from pandas import read_csv
import statsmodels.api as sm
import matplotlib.pyplot as plt
from math import *

if __name__ == '__main__':
    line = '______________________________________________________________________\n'
    values = read_csv('data.csv')

    # Вариационный ряд
    print(line)
    print('Набор исходных данных:')
    print(values)
    sorted_list = values.sort_values(by='VALUES')
    print(line)
    print('Вариационный ряд:')
    print(sorted_list)

    # Экстремальные значения и размах
    print(line)
    print('Экстремальные значения:\nX[1] = ' + str(sorted_list['VALUES'].iat[0]) + '\nX[n] = ' + str(
        sorted_list['VALUES'].iat[-1]) + '\n')
    scope = sorted_list['VALUES'].iat[-1] - sorted_list['VALUES'].iat[0]
    print('Размах: ' + str(scope))

    # Оценки мат. ожидания и среднеквадратического отклонения
    print(line)
    print('Математическое ожидание: ' + str(sorted_list['VALUES'].mean()))
    print('Среднеквадратическое отклонение: ' + str(sorted_list['VALUES'].std()))
    print(line)

    # График эмпирической функции распределения
    ecdf = sm.distributions.ECDF(sorted_list['VALUES'])
    print('Значения функции эмпирического распределения:\n' + str(ecdf.y))
    x = np.linspace(min(sorted_list['VALUES']), max(sorted_list['VALUES']))
    y = ecdf(x)
    plt.step(x, y, color='green')
    plt.title('Эмпирическая функция распределения')
    plt.show()
    print(line)

    # Гистограмма и график статистического распределения
    height = scope / (1 + log2(sorted_list['VALUES'].size))
    print('Значение h для гистограммы: ' + str(height))
    # Считаем интервалы
    bins = np.arange(sorted_list['VALUES'].iat[0], sorted_list['VALUES'].iat[-1], height)
    print('Интервалы гистограммы:\n' + str(bins))
    # Построение гистограммы
    histogram = sorted_list['VALUES'].hist(bins=bins, color='green', edgecolor='black')
    histogram.xaxis.set_tick_params(labelsize=8)
    # Маркировка середин интервалов
    middles = np.arange(sorted_list['VALUES'].iat[0] + height / 2, sorted_list['VALUES'].iat[-1], height)
    plt.xticks(middles)
    plt.title('Гистограмма')
    plt.show()

    # Полигон частот
    polygon = sorted_list['VALUES'].plot.kde(ind=bins.size, color='green')
    print(polygon.get_lines()[0].get_data())
    plt.title('Полигон частот')
    plt.show()
