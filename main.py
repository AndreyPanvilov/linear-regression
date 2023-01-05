import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats


def reader_csv(arr):
    with open("states.csv", encoding='utf-8') as r_file:
        # Создаем объект reader, указываем символ-разделитель ","
        file_reader = csv.reader(r_file, delimiter=",")
        # Счетчик для подсчета количества строк и вывода заголовков столбцов
        count = 0
        # Считывание данных из CSV файла
        for row in file_reader:
            if count == 0:
                # Вывод строки, содержащей заголовки для столбцов
                print(f'Файл содержит столбцы: {", ".join(row)}')
            else:
                arr.append([float(row[3]), float(row[4])])
            count += 1
        #print(f'Всего в файле {count} строк.')


def dispertion(arr):
    pass


def mean_y(arr):
    sumy = 0
    for i in range(len(arr)):
        sumy += arr[i][1]
    avgy = sumy / len(arr)
    return avgy


def mean_x(arr):
    sumx = 0
    for i in range(len(arr)):
        sumx += arr[i][0]
    avgx = sumx / len(arr)
    return avgx


def find_max_x(arr):
    max0 = 0
    for i in range(len(arr)):
        if max0 < arr[i][0]:
            max0 = arr[i][0]
    return max0


def find_min_x(arr):
    min0 = 100
    for i in range(len(arr)):
        if min0 > arr[i][0]:
            min0 = arr[i][0]
    return min0


def find_max_y(arr):
    max1 = 0
    for i in range(len(arr)):
        if arr[i][1] > max1:
            max1 = arr[i][1]
    return max1


def find_min_y(arr):
    min1 = 100
    for i in range(len(arr)):
        if arr[i][1] < min1:
            min1 = arr[i][1]
    return min1


def plot(arr):
    for i in range(len(arr)):
        plt.plot(arr[i][0], arr[i][1], 'ro')

    plt.axis([find_min_x(arr)-1, find_max_x(arr)+1, find_min_y(arr)-1, find_max_y(arr)+1])

    plt.title("Связь бедности и уровня образования", fontsize=14, fontweight="bold")
    plt.xlabel("Среднее образование(%)", fontsize=14, fontweight="bold")
    plt.ylabel("Бедность(%)", fontsize=14, fontweight="bold")

    plt.show()


def cor(arr):
    avgx = mean_x(arr)
    avgy = mean_y(arr)
    numerator, denumenator1, denumenator2 = 0, 0, 0
    for i in range(len(arr)):
        numerator += (arr[i][0] - avgx) * (arr[i][1] - avgy)
        denumenator1 += (arr[i][0] - avgx) * (arr[i][0] - avgx)
        denumenator2 += (arr[i][1] - avgy) * (arr[i][1] - avgy)
    r = round(numerator / (math.sqrt(denumenator1) * math.sqrt(denumenator2)), 2)
    return r


def find_b1(arr):
    avgx = mean_x(arr)
    avgy = mean_y(arr)
    sumx, sumy = 0, 0
    for i in range(len(arr)):
        sumx += (arr[i][0] - avgx) * (arr[i][0] - avgx)
        sumy += (arr[i][1] - avgy) * (arr[i][1] - avgy)
    dx = math.sqrt(sumx / (len(arr) - 1))
    dy = math.sqrt(sumy / (len(arr) - 1))
    b1 = dy/dx * cor(arr)
    return b1


def find_b0(arr):
    avgx = mean_x(arr)
    avgy = mean_y(arr)
    b0 = avgy - avgx * find_b1(arr)
    return b0


def plot_line(arr):
    c = find_b0(arr)
    b = find_b1(arr)
    y = lambda x: c + b * x
    max0, max1 = 0, 0
    min0, min1 = 100, 100
    for i in range(len(arr)):
        plt.plot(arr[i][0], arr[i][1], 'ro')

    plt.axis([find_min_x(arr)-1, find_max_x(arr)+1, find_min_y(arr)-1, find_max_y(arr)+1])

    plt.title("Связь бедности и уровня образования", fontsize=14, fontweight="bold")
    plt.xlabel("Среднее образование(%)", fontsize=14, fontweight="bold")
    plt.ylabel("Бедность(%)", fontsize=14, fontweight="bold")

    x = np.linspace(min0, max0, 100)
    plt.plot(x, y(x))
    plt.show()


def find_std_error_hs_grad(arr):
    #ищем среднее отклонение наших величин от регрессионной прямой
    s = 0
    b0 = find_b0(arr)
    b1 = find_b1(arr)
    for i in range(len(arr)):
        s += (arr[i][1] - (b0 + b1 * arr[i][0])) * (arr[i][1] - (b0 + b1 * arr[i][0]))
    s = s / (len(arr) - 2)
    avgx = mean_x(arr)
    sumx = 0
    for i in range(len(mas)):
        sumx += (mas[i][0] - avgx) * (mas[i][0] - avgx)
    sb = s / sumx
    return math.sqrt(sb)


def find_std_error_intercept(arr):
    sb = find_std_error_hs_grad(arr)
    sum = 0
    for i in range(len(arr)):
        sum += arr[i][0] * arr[i][0]
    sa = sb * math.sqrt(sum)
    return sa / math.sqrt(len(arr))


def find_t_value_b1(arr):
    return find_b1(arr) / find_std_error_hs_grad(arr)


def find_t_value_b0(arr):
    return find_b0(arr) / find_std_error_intercept(arr)


def F_critery(arr):
    f = cor(arr)*cor(arr) / (1 - cor(arr)*cor(arr)) * (len(arr) - 2)
    return f


if __name__ == '__main__':
    a = []
    reader_csv(a)
    mas = np.array(a)
    print('corelation:', cor(mas))
    print('R2 = ', cor(mas)*cor(mas))
    print('уравнение регрессионной прямой: y = ', round(find_b0(mas), 2), ' ', round(find_b1(mas), 2), 'x', sep='')
    print('std err:')
    print('b1 =', find_std_error_hs_grad(mas))
    print('b0 = ', find_std_error_intercept(mas))
    print("Фактическое значение t критерия Cтьюдента для b0:", find_t_value_b0(mas))
    print("Фактическое значение t критерия Cтьюдента для b1:", find_t_value_b1(mas))
    print('Вероятность получить такое или еще более выраженное отклонение для b0:',
          scipy.stats.t.sf(abs(find_t_value_b0(mas)), df=len(mas) - 2))
    print('Вероятность получить такое или еще более выраженное отклонение для b1:',
          scipy.stats.t.sf(abs(find_t_value_b1(mas)), df=len(mas) - 2))
    print('Эмпирическое значение критерия Фишера:', F_critery(mas))
    print('Критическое значение критерия Фишера:', scipy.stats.f.ppf(q=1-.05, dfn=1, dfd=len(mas) - 2))
    plot_line(mas)
