import pandas as pd
import glob
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

def make_box_plot(df, x_axis, y_axis, hue, title, font_size=14, height=5, aspect=3, width=0.8, palette=None, log=False, **kwargs):
    """
    Create a box plot using seaborn.
    
    Parameters:
    - df: DataFrame containing the data to plot.
    - x_axis: Column name for the x-axis.
    - y_axis: Column name for the y-axis.
    - hue: Column name for the hue (color) grouping.
    - title: Title of the plot.
    - font_size: Font size for the plot.
    - height: Height of the plot.
    - aspect: Aspect ratio of the plot.
    - width: Width of the boxes.
    - palette: Color palette for the plot.
    
    df must have columns:
    | x_axis | y_axis | hue |
    |--------|--------|-----|
    """
    plt.rc('font', family='serif', size=font_size)
    sns.catplot(data=df, x=x_axis, y=y_axis, hue=hue, kind="box", height=height, aspect=aspect, width=width, palette=palette)
    plt.xticks(rotation=90)
    if log:
        plt.yscale('log')
    plt.title(title)

def make_bar_plot(df, x_axis, y_axis, hue, title='', font_size=14, width=0.8, palette=None, log=False, **kwargs):
    """
    Create a bar plot using seaborn.

    Parameters:
    - df: DataFrame containing the data to plot.
    - x_axis: Column name for the x-axis.
    - y_axis: Column name for the y-axis.
    - hue: Column name for the hue (color) grouping.
    - title: Title of the plot.
    - font_size: Font size for the plot.
    - width: Width of the bars as a decimal - less than 1 gives more space between bars.
    - palette: Color palette for the plot as a dictionary.
    
    df must have columns:
    | x_axis | y_axis | hue |
    |--------|--------|-----|
    """
    plt.rc('font', family='serif', size=font_size)
    plt.figure()
    sns.barplot(df, x=x_axis, y=y_axis, hue=hue, width=width, palette=palette)
    plt.xticks(rotation=90)
    plt.title(title)

    if log:
        plt.yscale('log')
    plt.show()