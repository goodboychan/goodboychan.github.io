import glob
import io
import base64
import numpy as np
import pandas as pd
import seaborn as sns
np.set_printoptions(precision=2, suppress=True)

import matplotlib
import matplotlib.pyplot as plt

from IPython.display import HTML
from IPython import display as ipythondisplay


"""
Utility functions to enable video recording of gym environment and displaying it
To enable video, just do "env = wrap_env(env)""
"""


def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_model_params(model):
    return sum(p.numel() for p in model.parameters())


def plot_grid_std_learning_curves(d, num_it):
    for i, key in enumerate(d):
        ax = plt.subplot(2, 2, i+1)
        rewards, success_rates = d[key]
        plot_std_learning_curves(rewards, success_rates, num_it, no_show=True)
        ax.set_title(key)
    plt.show()


def plot_std_learning_curves(rewards, success_rates, num_it, no_show=False):
    r, sr = np.asarray(rewards), np.asarray(success_rates)
    df = pd.DataFrame(r).melt()
    sns.lineplot(x="variable", y="value", data=df, label='reward/eps')
    df = pd.DataFrame(sr).melt()
    sns.lineplot(x="variable", y="value", data=df, label='success rate')
    plt.xlabel("Training iterations")
    plt.ylabel("")
    plt.xlim([0, num_it])
    plt.ylim([0, 1])
    plt.legend()
    plt.grid('on')
    if not no_show:
        plt.show()


def plot_learning_curve(rewards, success_rate, num_it, plot_std=False):
    if plot_std:
        # plots shaded regions if list of reward timeseries is given
        plot_std_learning_curves(rewards, success_rate, num_it)
    else:
        plt.plot(rewards, label='reward/eps')
        if success_rate:
            plt.plot(success_rate, label='success rate')
            plt.legend()
        else:
            plt.ylabel('return / eps')
        plt.ylim([0, 1])
        plt.xlim([0, num_it - 1])
        plt.xlabel('train iter')
        plt.grid('on')
        plt.show()

# Wrapper around dictionary that enables attribute access instead of the bracket syntax
# i.e. you can replace d['item'] with d.item
class ParamDict(dict):
    __setattr__ = dict.__setitem__
    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)
    def __getstate__(self): return self
    def __setstate__(self, d): self = d

