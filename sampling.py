import pandas as pd 
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt


def import_data(file_path):
    
    df = pd.read_csv(file_path)
    df = df.dropna()
    return df


def 

def plot_data(df):
    O2_release = df['O2 Released (µmol)']
    temperature = df['calibrated temperature (C)']
    time = df["Time (s)"]

    plt.plot(time, O2_release)

    plt.tight_layout()
    plt.show()
    

def main(file_path):

    df = import_data(file_path)
    plot_data(df)
    return df
  




if __name__ == "__main__":
        file_path = "M2_2.18mM_pressure_o2_release.csv"
        main(file_path)
