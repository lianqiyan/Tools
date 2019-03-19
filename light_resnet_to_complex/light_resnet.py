import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# sns.set(style="whitegrid")

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

if __name__ == '__main__':
    # dat = pd.read_csv('16step2000_error.csv')
    # dat = pd.read_csv('32step2000_error.csv')
    dat = pd.read_csv('model16_32_64_128.csv')
    print(dat.head())
    ac_one = 0.1*np.ones(dat.shape[0])
    plt.plot(dat['step'], smooth(dat['train_error'].values,3), '--g')
    plt.plot(dat['step'],smooth(dat['validation_error'].values,3), 'r')
    plt.plot(dat['step'], ac_one, ':k')
    # plt.text(0, 0.06,'10% error')
    plt.legend(['train_error', 'test_error', '10% error line'])
    # ax.yaxis.set_ticks_position('left')
    # ax.xaxis.set_ticks_position('bottom')
    plt.xlabel('Epoch')
    plt.ylabel('error rate')
    # sns.lineplot(x='step', y='train_error', data=dat,  palette="ch:2.5,.25", linewidth=1.5)
    # sns.lineplot(x='step', y='validation_error', data=dat,  palette="ch:2.5,.25", linewidth=1.5)

    plt.show()


