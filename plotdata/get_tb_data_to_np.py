import csv
import os
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt




def plot_tensorboard_log(logdir, dir, plot=False):
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()

    all_steps = []
    all_values = []

    # 获取指定tag的数据
    for tag in event_acc.Tags()['scalars']:
        print(tag)
        scalars = event_acc.Scalars(tag)
        steps = [scalar.step for scalar in scalars]
        values = [scalar.value for scalar in scalars]

        with open("datas/{}-{}.csv".format(dir, str(tag).replace("/", "-")), "w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(['steps', 'values'])
            for step, value in zip(steps, values):
                writer.writerow([step, value])

        if plot:
            plt.plot(steps, values)
            plt.xlabel('Steps')
            plt.ylabel(tag)
            plt.title('TensorBoard Log - {}'.format(tag))
            plt.grid(True)
            plt.show()
    else:
        print("Tag '{}' not found in TensorBoard log.".format(tag))

if __name__ == "__main__":
    for i in range(10):
        logdir = r'D:\codes\python\spacegroup_classification\result\{}\model_results\tensorboard'.format(i)

        plot_tensorboard_log(logdir, i)
