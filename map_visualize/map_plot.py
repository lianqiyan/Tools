from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_data():
    dt = pd.read_excel('lat_long.xlsx')
    point= dt.values
    name = list(dt.index)
    crs = open("lat_long2.txt", "r")
    pair = []
    for columns in (raw.strip().split() for raw in crs):
        if len(columns)< 1:
            break
        else:
            strnum = columns[2].strip('(, )').split(',')
            num = [int(i) for i in strnum]
            # print(num)
            pair.append(num)
    return point, name, pair


def show_map(dat, na, lp):
    fig = plt.figure(figsize=(32, 16))
    # m = Basemap(width=12000000,height=9000000, projection='lcc', resolution='c',
    #             lat_0=38, lon_0=96, )

    m = Basemap(llcrnrlon=77, llcrnrlat=14, urcrnrlon=140, urcrnrlat=51, projection='lcc', lat_1=33, lat_2=45,
                lon_0=100)

    m.etopo(scale=0.5, alpha=0.5)

    for i in range(0, dat.shape[0]):
        x, y = m(dat[i, 0], dat[i, 1])
        plt.plot(x, y, 'ok', markersize=1)
        plt.text(x, y, na[i], fontsize=3)

    for i in range(0, len(lp)):
        # print(lp[i])
        pai = [ nu-1 for nu in lp[i]]
        lon = [dat[pai[0], 0], dat[pai[1], 0]]
        lat = [dat[pai[0], 1], dat[pai[1], 1]]

        # print(pai)
        # print(lat, lon)
        x, y = m(lon, lat)
        m.plot(x, y, 'o-', markersize=3, linewidth=1)

    # lat = [21.27, 24.24]
    # lon = [110.36, 118.09]



    # m.drawcoastlines()
    # m.fillcontinents(color='white')
    # m.drawmapboundary(fill_color='white')
    # m.drawstates(color='black')
    # m.drawcountries(color='black')
    plt.title("#wedgez")
    plt.show()
    fig.savefig('test.png', dpi=500)


    # Map (long, lat) to (x, y) for plotting
    # x, y = m(114.06, 22.54)
    # plt.plot(x, y, 'ok', markersize=2)
    # plt.text(x, y, ' Shenzhen', fontsize=15)
    # plt.show()


if __name__ == '__main__':
    dot, name, dp = read_data()
    show_map(dot, name, dp)
