import os
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

def plot_tensorboard_log(logdir, tag):
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()

    # 获取指定tag的数据
    if tag in event_acc.Tags()['scalars']:
        scalars = event_acc.Scalars(tag)
        steps = [scalar.step for scalar in scalars]
        values = [scalar.value for scalar in scalars]

        plt.plot(steps, values)
        plt.xlabel('Steps')
        plt.ylabel(tag)
        plt.title('TensorBoard Log - {}'.format(tag))
        plt.grid(True)
        plt.show()
    else:
        print("Tag '{}' not found in TensorBoard log.".format(tag))

if __name__ == "__main__":
    logdir = r'..\spacegroup_classification\RCNet\model_results\tensorboard'
    tag = 'Loss/Train'

    plot_tensorboard_log(logdir, tag)
