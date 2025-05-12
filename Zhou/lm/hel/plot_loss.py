import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import random
import matplotlib.colors as mcolors
def generate_distinguishable_colors(n):
    colors = []
    for i in range(n):
        # 生成均匀分布的色相值
        hue = i / n
        # 使用固定的饱和度和亮度生成颜色
        color = mcolors.hsv_to_rgb((hue, 0.7, 0.9))
        # 将 RGB 颜色转换为十六进制颜色
        hex_color = mcolors.to_hex(color)
        colors.append(hex_color)
    return colors

def log_formatter(x, pos):
    return f'$10^{{{int(x)}}}$'

def plot_losses(Loss, labels, colors, linestyles,time=None, line_widths=1.5, min_loss_offsets=0.1, title="Training GPU time", xlabel="time sec.", ylabel="Loss", figsize=(7, 6)):
    """
    绘制损失曲线，并标注每条曲线的最小损失值。

    参数:
    - loss: tuple  of np.array, 包含loss的元组
    - labels: list of str, 每条曲线的标签
    - colors: list of str, 每条曲线的颜色
    - linestyles: list of str, 每条曲线的线型
    - line_widths: list of float, 每条曲线的线宽
    - min_loss_offsets: list of float or None, 每条曲线的最小损失值标注偏移量
    - title: str, 图表标题
    - xlabel: str, x轴标签
    - ylabel: str, y轴标签
    - figsize: tuple, 图表大小
    """
    figure = plt.figure(figsize=figsize)
    ax = figure.add_subplot(1, 1, 1)

    for i, l in enumerate(Loss):
        loss = Loss[i]
        if np.any(loss != 0):
            loss = np.log10(loss[loss != 0])
        else:
            continue
        if time is not None:
            Time=time[i]
            ax.plot(Time[0:len(loss)],loss, color=colors[i], label=labels[i], linestyle=linestyles, linewidth=line_widths)
        else:
            ax.plot(loss, color=colors[i], label=labels[i], linestyle=linestyles, linewidth=line_widths)

        

    ax.set_title(title, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    
    ax.yaxis.set_major_formatter(FuncFormatter(log_formatter))
    ax.legend(fontsize=12)
    
    plt.show()

# 使用示例
'''
plot_losses(
    record_files=[
        "losspinns_record98.npy",
        "losspinnsall_record99.npy",
        "losspinns_record97.npy",
        "Burgers1.npy"
    ],
    labels=[
        "Gradual increase",
        "LM",
        "Random Choose",
        "PINN"
    ],
    colors=['r', 'b', 'y', 'g'],
    linestyles=['--', '--', '--', '--'],
    line_widths=[2, 2, 2, 2],
    min_loss_offsets=[0.1, -0.4, -0.2, 0],
    title="Loss Curve",
    xlabel="iteration",
    ylabel="loss"
)
'''


