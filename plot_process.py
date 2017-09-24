# -*- coding: utf-8 -*-
import numpy as np
import re
import click
from matplotlib import pylab as plt


@click.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True))
def main(files):
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('loss')
    #读取log文件并绘图
    for i, log_file in enumerate(files):
        loss_iterations, losses, test_iterations, test_losses = parse_log(log_file)
        disp_results(fig, ax1, loss_iterations, losses, color_ind=i)
        #disp_results(fig, ax1, test_iterations, test_losses, color_ind=i+1)
    plt.show()

# 解析log文件
def parse_log(log_file):
    with open(log_file, 'r') as log_file:
        log = log_file.read()

    # iter_num: group0, loss_val: group1
    # Iteration iter_num(\d), loss = loss_val +-((0(.0))or(.0))(eE+-0)
    loss_pattern = r"Iteration (?P<iter_num>\d+), loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    test_num_pattern = r"Iteration (?P<test_num>\d+), Testing net"
    test_loss_pattern = r"Test net output #0: loss = (?P<test_loss>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    losses = []
    loss_iterations = []
    test_losses = []
    test_iterations = []

    for r in re.findall(loss_pattern, log):
        loss_iterations.append(int(r[0])) # i.e. iter_num
        losses.append(float(r[1])) # i.e. loss_val

    for iter_num in re.findall(test_num_pattern, log):
        test_iterations.append(int(iter_num))

    for loss_val in re.findall(test_loss_pattern, log):
        test_losses.append(float(loss_val[0]))

    loss_iterations = np.array(loss_iterations)
    losses = np.array(losses)

    test_iterations = np.array(test_iterations)
    test_losses = np.array(test_losses)

    return loss_iterations, losses, test_iterations, test_losses


def disp_results(fig, ax1, loss_iterations, losses, color_ind=0):
    modula = len(plt.rcParams['axes.color_cycle'])
    ax1.plot(loss_iterations, losses, color=plt.rcParams['axes.color_cycle'][(color_ind * 2 + 0) % modula])

if __name__ == '__main__':
    main()