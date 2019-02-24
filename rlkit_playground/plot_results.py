import glob
from datetime import timedelta

import _rlkit
from rlkit.launchers.config import LOCAL_LOG_DIR
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot(exp_name, local_dir=None):
    local_dir = local_dir or LOCAL_LOG_DIR
    path_format = '{local_dir}/{exp_name}/*/progress.csv'.format(local_dir=local_dir, exp_name=exp_name)
    paths = glob.glob(path_format)

    for p in paths:
        run_name = p.split('/')[-2]
        df = pd.read_csv(p)
        steps_per_epoch = df['Number of env steps total'][0]

        fig, ax1 = plt.subplots()
        y = 1 + df['Test Rewards Mean'].values
        ax1.plot(df['Epoch'].values, y)
        ax1.set_ylabel('Success rate')
        ax1.set_xlabel('Epoch ({} env. steps each)'.format(steps_per_epoch))

        ax2 = ax1.twiny()
        ax2.plot(df['Total Train Time (s)'].values, y).pop(0).remove()

        def format_date(x, pos=None):
            return str(timedelta(seconds=x))
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        for tick in ax2.get_xticklabels():
            tick.set_rotation(45)

        plt.title(run_name)
        plt.show()


if __name__ == '__main__':
    plot('mordor/her-tsac-fetch-pp')
    plot('mordor/her-td3-fetch-pp')


