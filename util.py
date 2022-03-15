import matplotlib.pyplot as plt
import seaborn as sns

res1 = {1: 2.6999999999943736e-06, 2: 2.189999999999137e-05, 3: 0.0001249000000000111, 4: 0.0009548999999999808,
        5: 0.0130247, 6: 0.059406599999999976, 7: 0.4865899, 8: 3.8112233999999994, 9: 31.138999300000002,
        10: 248.59914579999997}
res2 = {1: (4, 8), 2: (40, 64), 3: (336, 512), 4: (2720, 4096), 5: (21824, 32768), 6: (174720, 262144),
        7: (1398016, 2097152), 8: (11184640, 16777216), 9: (89478144, 134217728), 10: (715827200, 1073741824)}


def plot_time(data: dict, title: str):
    _x = list(data.keys())
    _y = [data[key] for key in data.keys()]
    sns.set_style("darkgrid")
    plt.plot(_x, _y, lw=2, marker='o')
    # plt.legend(title='Binet', loc='upper left', labels=['Hell Yeh', 'Nah Bruh'])

    plt.title(title)
    plt.xlabel('I: size of matrix 2**I')
    plt.ylabel('Time [s]')

    plt.yscale('log')
    plt.show()


def plot_counters(data, title: str):
    _x = list(data.keys())
    _y1 = [data[key][0] for key in data.keys()]
    _y2 = [data[key][1] for key in data.keys()]

    sns.set_style("darkgrid")
    plt.plot(_x, _y1, lw=2, marker='o')
    plt.plot(_x, _y2, lw=2, marker='o')
    plt.legend(title='Binet', loc='upper left', labels=['Multiplications', 'Additions'])

    plt.title(title)
    plt.xlabel('I: size of matrix 2**I')
    plt.ylabel('Time [s]')

    plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    plot_time(res1, title='Binet')
    plot_counters(res2, title='Binet')
