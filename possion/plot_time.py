import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MultipleLocator
def log_formatter(x, pos):
    return f'$10^{{{int(x)}}}$'

def ms_formatter(x, pos):
    return f'{x:.3f}ms'

def plot_time(time_arrary, x_axis,color, linestyle, line_widths=1.5, min_loss_offsets=0, title="avg elapsed time", xlabel="parameter numbers", ylabel="ms", figsize=(5, 5)):
    """s
    绘制时间曲线，并标注每条曲线的最小损失值。
    参数:
    - time_array: numpy array, 包含平均时间的曲线
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

    #Time
    ax2 = figure.add_subplot(1, 1,1)
    time = time_arrary
    time =time[time != 0]
    
    
    ax2.plot(x_axis,time, color, label='time', linestyle=linestyle, linewidth=line_widths)
    
    min_time = np.min(time)
    min_offset = min_loss_offsets if min_loss_offsets else 0
    
    ax2.annotate(f'min time: {min_time:.2f}', xy=(x_axis[np.argmin(time)], min_time),
                xytext=(x_axis[np.argmin(time)], min_time + min_offset),
                arrowprops=dict(facecolor=color, shrink=0.05))

    ax2.set_title(title, fontsize=16)
    ax2.set_ylabel(ylabel, fontsize=14)
    ax2.set_xlabel(xlabel, fontsize=14)
    ax2.xaxis.set_major_locator(MultipleLocator(100))
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.yaxis.set_major_formatter(FuncFormatter(ms_formatter))
    ax2.legend(fontsize=12)
    
    plt.show()

# 使用示例
'''
'''