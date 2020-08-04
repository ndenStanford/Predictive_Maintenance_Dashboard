import pandas as pd
import numpy as np

#import matplotlib
#import seaborn
#import matplotlib.dates as md
#from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
#from pyemma import msm # not available on Kaggle Kernel
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

import timesynth as ts
from alibi_detect.od import SpectralResidual
from alibi_detect.utils.perturbation import inject_outlier_ts
from alibi_detect.utils.saving import save_detector, load_detector
from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_ts

from alibi_detect.od import OutlierSeq2Seq
from alibi_detect.utils.perturbation import inject_outlier_ts
from alibi_detect.utils.saving import save_detector, load_detector
from alibi_detect.utils.visualize import plot_feature_outlier_ts, plot_roc

import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow.keras.layers import Dense, InputLayer
from alibi_detect.datasets import fetch_kdd
from alibi_detect.models.losses import elbo
from alibi_detect.od import OutlierVAE
from alibi_detect.utils.data import create_outlier_batch
from alibi_detect.utils.fetching import fetch_detector
from alibi_detect.utils.saving import save_detector, load_detector
from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_tabular, plot_roc

def preprocess_data(df):
    # change the type of timestamp column for plotting
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # change fahrenheit to °C (temperature mean= 71 -> fahrenheit)
    df['value'] = (df['value'] - 32) * 5/9
    # the hours and if it's night or day (7:00-22:00)
    df['hours'] = df['timestamp'].dt.hour
    df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)
    # the day of the week (Monday=0, Sunday=6) and if it's a week end day or week day.
    df['DayOfTheWeek'] = df['timestamp'].dt.dayofweek
    df['WeekDay'] = (df['DayOfTheWeek'] < 5).astype(int)
    # time with int to plot easily
    df['time_epoch'] = (df['timestamp'].astype(np.int64)/100000000000).astype(np.int64)
    # creation of 4 distinct categories that seem useful (week end/day week & night/day)
    df['categories'] = df['WeekDay']*2 + df['daylight']
    # Take useful feature and standardize them 
    data = df[['value', 'hours', 'daylight', 'DayOfTheWeek', 'WeekDay']]
    min_max_scaler = preprocessing.StandardScaler()
    np_scaled = min_max_scaler.fit_transform(data)
    data = pd.DataFrame(np_scaled)
    return data
    
def isolation_forest(data,df):
    # An estimation of anomly population of the dataset (necessary for several algorithm)
    outliers_fraction = 0.01
    # train isolation forest 
    model =  IsolationForest(contamination = outliers_fraction)
    model.fit(data)
    # add the data to the main  
    df['anomaly_isf'] = pd.Series(model.predict(data))
    df['anomaly_isf'] = df['anomaly_isf'].map( {1: 0, -1: 1} )
    a = df.loc[df['anomaly_isf'] == 1, ['timestamp', 'value']] #anomaly
    return a
    
def one_class_svm(data,df):
    # An estimation of anomly population of the dataset (necessary for several algorithm)
    outliers_fraction = 0.01
    # train one class SVM 
    model =  OneClassSVM(nu=0.95 * outliers_fraction) #nu=0.95 * outliers_fraction  + 0.05
    model.fit(data)
    # add the data to the main  
    df['anomaly_svm'] = pd.Series(model.predict(data))
    df['anomaly_svm'] = df['anomaly_svm'].map( {1: 0, -1: 1} )
    a = df.loc[df['anomaly_svm'] == 1, ['timestamp', 'value']] #anomaly
    return a

def spectral_residual(df):    
    od = SpectralResidual(
        threshold=None,  # threshold for outlier score
        window_amp=20,   # window for the average log amplitude
        window_local=20, # window for the average saliency map
        n_est_points=20  # nb of estimated points padded to the end of the sequence
    )
    X_outlier=np.array(df['value'])
    time_samples=np.arange(0,len(df))
    od.infer_threshold(X_outlier, time_samples, threshold_perc=99)
    od_preds = od.predict(X_outlier, time_samples, return_instance_score=True)
    pred=(od_preds['data']['is_outlier']==1)
    a = df.loc[pred]
    return a

def seq2seq(data,df):
    X_train=np.array(data).astype(np.float32)
    # initialize outlier detector
    n_features=5
    seq_len=50
    od = OutlierSeq2Seq(n_features,
                        seq_len,
                        threshold=None,
                        latent_dim=20)
    # train
    od.fit(X_train[:7200],
           epochs=10,
           verbose=False)
    od.infer_threshold(X_train[:7200], threshold_perc=95)
    od_preds = od.predict(X_train[:7200],
                      outlier_type='instance',    # use 'feature' or 'instance' level
                      return_feature_score=True,  # scores used to determine outliers
                      return_instance_score=True)
    pred=(od_preds['data']['is_outlier']==1)
    a=(df.iloc[:7200]).loc[pred]
    return a

