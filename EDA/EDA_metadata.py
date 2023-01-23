import seaborn as sns
import matplotlib.pyplot as plt


def plot_multiples(df, sample_times, ylim = 1e7):
    select = df[df.lat_long.isin((df['lat_long'].value_counts()>=sample_times).index[(df['lat_long'].value_counts()>=sample_times)])].copy()# make df containing only those
    sns.set(rc={'figure.figsize':(16,5)})
    fig = plt.figure()
    ax = plt.axes()
    pal = sns.color_palette("Paired", select.lat_long.nunique())
    for i,place in enumerate(select.lat_long.unique()):
        sns.lineplot(y=select[select.lat_long == place].density, x = select[select.lat_long == place].date, color = pal[i])
        sns.scatterplot(y=select[select.lat_long == place].density, x = select[select.lat_long == place].date, color = pal[i])
    ax.set_ylim(0,ylim)

    