import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# from sklearn.preprocessing import PolynomialFeatures
from scipy import interpolate

# res1 = {1: 2.6999999999943736e-06, 2: 2.189999999999137e-05, 3: 0.0001249000000000111, 4: 0.0009548999999999808,
#         5: 0.0130247, 6: 0.059406599999999976, 7: 0.4865899, 8: 3.8112233999999994, 9: 31.138999300000002,
#         10: 248.59914579999997, 11: 1996.5359772}
# res2 = {1: (4, 8), 2: (40, 64), 3: (336, 512), 4: (2720, 4096), 5: (21824, 32768), 6: (174720, 262144),
#         7: (1398016, 2097152), 8: (11184640, 16777216), 9: (89478144, 134217728), 10: (715827200, 1073741824),
#         11: (5726621696, 8589934592)}

res1_binet = {1: 4.300000000068138e-06, 2: 3.320000000006651e-05, 3: 0.00021209999999993734, 4: 0.0017792999999999282,
              5: 0.012322999999999862, 6: 0.10045579999999998, 7: 0.8102378000000001, 8: 6.4519477, 9: 52.595525,
              10: 431.07566049999997, 11: 3585.0909906}

res2_binet = {1: (0, 0), 2: (48, 80), 3: (528, 800), 4: (4556, 6736), 5: (36800, 54400), 6: (293888, 436480),
              7: (2343168, 3494400), 8: (18699276, 27960336), 9: (149376000, 223692800), 10: (1194057728, 1789562880),
              11: (9548484608, 14316544000)}

res1_str = {1: 5.199999999927485e-06, 2: 0.00029180000000006423, 3: 0.002840899999999813, 4: 0.020215799999999895,
            5: 0.17024430000000002, 6: 1.1075860000000002, 7: 7.942406, 8: 54.4047946, 9: 395.51187300000004,
            10: 2681.0317936}

res2_str = {1: (0, 0), 2: (188, 70), 3: (2388, 630), 4: (21684, 4704), 5: (172500, 33390), 6: (1294028, 234850),
            7: (9409428, 1646190), 8: (67281204, 11527824), 9: (476649300, 80703630),
            10: (3359310668, 564943330)}

lab03_time_binet = {1: 5.699999999997374e-06, 2: 3.3699999999914354e-05, 3: 0.00022230000000011962,
                    4: 0.0015115999999999463,
                    5: 0.010312700000000063, 6: 0.08394950000000012, 7: 0.6656057000000002, 8: 5.3861151, 9: 43.6830945,
                    10: 362.46278159999997}

lab03_ops_binet = {1: (1, 2), 2: (27, 46), 3: (365, 570), 4: (3555, 5342), 5: (30000, 44512), 6: (243746, 361668),
                   7: (1953336, 2907592), 8: (15598758, 23288636), 9: (124584704, 186383872),
                   10: (995574272, 1491241984)}

lab03_time_str = {1: 5.699999999997374e-06, 2: 0.0004481999999998987, 3: 0.0019671999999999468, 4: 0.016265500000000044,
                  5: 0.12381819999999988, 6: 0.9031782000000002, 7: 6.434086700000001, 8: 46.114228,
                  9: 322.75968450000005}

lab03_ops_str = {1: (1, 2), 2: (97, 41), 3: (1575, 465), 4: (16455, 3942), 5: (142088, 29611), 6: (1100872, 209246),
                 7: (8200210, 1476323), 8: (59450525, 10362778), 9: (424506677, 72605113)}

lab03_det = {1: (1.0, 12.0), 2: (1.0, 31.999999999999986), 3: (0.9999999999999997, -20186.650269373393),
             4: (1.0000000000000009, 20518010950.203854), 5: (0.9666666666667205, 6.180725444831222e+30),
             6: (1.0000000000000355, 2.1229171590962987e+68), 7: (1.0000000000005587, 1.4454589455471688e+153),
             8: (1.0000000000016538, "inf")}


def plot_time(data: dict, data2: dict, title: str):
    _x = list(data.keys())
    _x2 = list(data2.keys())
    _y = [data[key] for key in data.keys()]
    _y2 = [data2[key] for key in data2.keys()]
    sns.set_style("darkgrid")
    plt.plot(_x, _y, lw=2, marker='o')
    # plt.legend(title='Binet', loc='upper left', labels=['Hell Yeh', 'Nah Bruh'])
    plt.plot(_x2, _y2, lw=2, marker='x')

    # lm = PolynomialFeatures(2)
    # lm.fit_transform(np.array(_x).reshape(-1, 1), _y)
    #
    # # test_y = [lm.predict(xx) for xx in test_x]
    # test_y = lm.transform(test_x)
    # np.polyfit(_x[1:], _y[1:], 3)
    #
    # test_x = np.arange(2, 20).reshape(-1, 1)
    # plt.plot(test_x, np.poly3d(test_x), lw=0.5)

    # f = interpolate.interp1d(_x[1:], _y[1:], fill_value="extrapolate", kind=9)
    # test_x = np.arange(2, 20)
    # test_y = f(test_x)

    # coef = np.polyfit([_x[1], _x[-1]], [_y[1], _y[-1]], 1)
    # poly = np.poly1d(coef)
    # test_x = np.arange(2, 20)
    # plt.plot(test_x, poly(test_x), lw=0.5)

    plt.title(title)
    plt.legend(title='Algorytm', loc='upper left', labels=['Binet', 'Strassen'])
    plt.xlabel('I: size of matrix 2**I')
    plt.ylabel('Time [s]')

    plt.yscale('log')
    plt.show()


def generate_charts(x_axis, y_time, y_multi_operations, y_addition_operations):
    plt.scatter(x_axis[:], y_time[:], color='b', label='Strassen algorithm')
    plt.plot(x_axis[:], y_time[:], color='b')
    plt.legend(loc="upper left")
    plt.xlabel('Matrix size')
    plt.ylabel('Multiplication time in s')
    plt.title('Time of multiplication by size of matrix')
    plt.grid()
    plt.show()

    plt.scatter(x_axis[:], y_multi_operations[:], color='r', label='Multiplication operations')
    plt.plot(x_axis[:], y_multi_operations[:], color='r')
    plt.xlabel('Matrix size')
    plt.ylabel('Amount of multiplication operations')
    plt.title('Amount of multiplication operations by size of matrix')
    plt.grid()
    plt.show()

    plt.scatter(x_axis[:], y_addition_operations[:], color='b', label='Addition operations')
    plt.plot(x_axis[:], y_addition_operations[:], color='b')
    plt.xlabel('Matrix size')
    plt.ylabel('Amount of addition operations')
    plt.title('Amount of addition operations by size of matrix')
    plt.grid()
    plt.show()


def plot_counters(data, data2, title: str):
    _x = list(data.keys())
    _y = [data[key][0] + data[key][1] for key in data.keys()]

    _x2 = list(data2.keys())
    _y2 = [data2[key][0] + data2[key][1] for key in data2.keys()]
    # _y2 = [data[key][1] for key in data.keys()]

    sns.set_style("darkgrid")
    plt.plot(_x, _y, lw=2, marker='o')
    # plt.plot(_x, _y2, lw=2, marker='o')
    # plt.legend(title='Binet', loc='upper left', labels=['Multiplications', 'Additions'])
    plt.plot(_x2, _y2, lw=2, marker='x')

    plt.title(title)
    plt.legend(title='Algorytm', loc='upper left', labels=['Binet', 'Strassen'])
    plt.xlabel('I: wielkość macierzy 2**I')
    plt.ylabel('Numer operacji')

    plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    plot_time(lab03_time_binet, lab03_time_str, title='Czas wykonywania programu')
    plot_counters(lab03_ops_binet, lab03_ops_str, title='Liczba operacji')
    # plot_counters(res2_str, title='Ilość operacji - Strassen')
