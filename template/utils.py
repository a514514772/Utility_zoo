'''
Misc Utility functions
'''
from collections import OrderedDict
import os
import numpy as np
import torch
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class LoggerUnit(object):
    def __init__(self):
        self.value_list = []
        self.step_list = []
        self.internal_step = 0

    def update(self, val, step=None):
        self.value_list.append(val)

        if step is not None:
            self.internal_step = step
            self.step_list.append(self.internal_step)
        else:
            self.step_list.append(self.internal_step)
        self.internal_step += 1

    def reset(self):
        self.__init__()

    def __getattr__(self, name):
        if name == 'val':
            return self.value_list
        elif name == 'step':
            return self.step_list
        elif name == 'sum':
            return np.sum(self.value_list)
        elif name == 'mean':
            return np.mean(self.value_list)

    def __setattr__(self, key, value):
        if key in ['value_list', 'step_list', 'internal_step']:
            super(LoggerUnit, self).__setattr__(key, value)
        else:
            return False


class Logger(object):
    def __init__(self, log_dir):
        self.dict = {}
        self.log_dir = log_dir
        mkdirs(self.log_dir)

    def __getitem__(self, key):
        return self.dict[key]

    def add_attribute(self, key):
        self.dict[key] = LoggerUnit()

    def update(self, key, value, step=None):
        if not key in self.dict:
            self.add_attribute(key)
        self.dict[key].update(value, step)

    def reset(self, key):
        if key in self.dict:
            self.dict[key].reset()

    def reset_all(self):
        for k in self.dict:
            self.dict[k].reset()

    def plot_curve(self):
        for key in self.dict:
            plt.plot(self.dict[key].val)
            plt.savefig(os.path.join(self.log_dir, key+'.jpg'))
            plt.close()



def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]


def poly_lr_scheduler(base_lr, iter, max_iter=30000, power=0.9):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(opts, base_lr, i_iter, max_iter, power):
    lr = poly_lr_scheduler(base_lr, i_iter, max_iter, power)
    for opt in opts:
        opt.param_groups[0]['lr'] = lr
        if len(opt.param_groups) > 1:
            opt.param_groups[1]['lr'] = lr * 10


def alpha_blend(input_image, segmentation_mask, alpha=0.5):
    """Alpha Blending utility to overlay RGB masks on RBG images
        :param input_image is a np.ndarray with 3 channels
        :param segmentation_mask is a np.ndarray with 3 channels
        :param alpha is a float value

    """
    blended = np.zeros(input_image.size, dtype=np.float32)
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    return blended


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state

    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def save_models(model_dict, prefix='./'):
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    for key, value in model_dict.items():
        torch.save(value.state_dict(), os.path.join(prefix, key + '.pth'))


def load_models(model_dict, prefix='./'):
    for key, value in model_dict.items():
        value.load_state_dict(torch.load(os.path.join(prefix, key + '.pth')))


def grid2gif(image_str, output_gif, delay=100):
    """Make GIF from images.

    code from:
        https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python/34555939#34555939
    """
    str1 = 'convert -delay ' + str(delay) + ' -loop 0 ' + image_str + ' ' + output_gif
    subprocess.call(str1, shell=True)


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    log = Logger()
    log.update('key', 123)
    log.update('a', 789)
    print (log['key'].val, log['key'].step)