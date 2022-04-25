import numpy as np
from pandas import read_csv
import statsmodels.api as sm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    line = '_________________________\n'
    values = read_csv('data.csv', ",")

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
    print('Размах: ' + str(sorted_list['VALUES'].iat[-1] - sorted_list['VALUES'].iat[0]))

    # Оценки мат. ожидания и среднеквадратического отклонения
    print(line)
    print('Математическое ожидание: ' + str(sorted_list['VALUES'].mean()))
    print('Среднеквадратическое отклонение: ' + str(sorted_list['VALUES'].std()))

    # График эмпирической функции распределения
    ecdf = sm.distributions.ECDF(sorted_list['VALUES'])
    x = np.linspace(min(sorted_list['VALUES']), max(sorted_list['VALUES']))
    y = ecdf(x)
    plt.step(x, y, color='green')
    plt.title('Эмпирическая функция распределения')
    plt.show()

    # Гистограмма и график статистического распределения
    histogram = sorted_list['VALUES'].hist(bins=20, color='green', edgecolor='black')
    sorted_list.plot(kind='kde', ax=histogram, secondary_y=True)
    plt.title('Графическое изображение статистического распределения')
    plt.show()
