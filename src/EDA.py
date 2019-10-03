import matplotlib as plt
import seaborn as sns

def plot_hist(df,saving_path):
    plt.figure(figsize = (5,5))
    df['Class'].hist(bins=3, density=True)
    plt.title(print(round((df['Class']==1).sum()/(len(df['Class']))*100,2),"% percentage of Class equals to 1 in the dataset"))
    plt.savefig(saving_path)
    return(plt.show())

def plot_heatmap(df,saving_path):
    plt.figure(figsize = (10,10))
    sns.heatmap(df.corr(), annot=False, linewidths=.5)
    plt.savefig(saving_path)
    return(plt.show())

def plot_time(df,saving_path):
    df['Time'].hist()
    plt.savefig(saving_path)
    return(plt.show())

def plot_amount(df,saving_path):
    sns.lmplot(x="Amount", y="Class", data=df)
    plt.savefig(saving_path)
    return(plt.show())