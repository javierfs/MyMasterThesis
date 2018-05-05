# Predict the Diagnose of Heartdisease
__author__ = 'Javier Fernandez'

import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('TkAgg')
import pylab as plt
from fancyimpute import MICE
import sys
# sys.path.append('/custom/path/to/modules')
import random
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import skew
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LassoCV
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn_pandas import DataFrameMapper
import xgboost as xgb
from matplotlib.backends.backend_pdf import PdfPages
import datetime
# from sklearn.cluster import FeatureAgglomeration
import seaborn as sns
# import math
#importing all the required ML packages
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix


class ClinicalData(object):
    def __init__(self):
        self.df = ClinicalData.df
        self.df_all_feature_var_names = []
        self.df_test_all_feature_var_names = []
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss')


    # Private variables
    _non_numerical_feature_names = []
    _numerical_feature_names = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    _categorical_feature_names = ['sex', 'cp', 'fbs', 'restecg','exang', 'slope','heartdisease']
    _is_one_hot_encoder = 0
    _feature_names_num = []
   # _save_path = '/home/mizio/Documents/Kaggle/HousePrices/prepared_data_train_and_test_saved/'
   # _is_not_import_data = 0
   # _is_dataframe_with_sale_price = 1

    ''' Pandas Data Frame '''
    filename = 'TotalDataBase.pkl'
    df = pd.read_pickle(filename)

    @staticmethod
    def extract_numerical_features(df):
        df = df.copy()
        # Identify numerical columns which are of type object
        numerical_features = pd.Series(data=False, index=df.columns, dtype=bool)

        for feature in df.columns:
            if any(tuple(df[feature].apply(lambda x: type(x)) == int)) or \
                            any(tuple(df[feature].apply(lambda x: type(x)) == float)) & \
                            (not any(tuple(df[feature].apply(lambda x: type(x)) == str))):
                numerical_features[feature] = 1
        return numerical_features[numerical_features == 1].index

    def clean_data(self, df):
        # 1 - Impute numerical features
        # 2 - Categorize numerical features
        # 3 - Impute categorical features
        # 4 - OHE
        df = df.copy()
        numerical_features_names = self._numerical_feature_names
        categorical_feature_names = self._categorical_feature_names
        if df[numerical_features_names].isnull().sum().sum() > 0:
            # Imputation using MICE
            df.loc[:, tuple(numerical_features_names)] = self.estimate_by_mice(df[numerical_features_names], 0, 0)
        #self.categorize_numerical_values(df)
        if df[categorical_feature_names].isnull().sum().sum() > 0:
            # Imputation using MICE sex,fbs,exang,resteg,slope
            _1st_group = ['sex','fbs', 'exang']
            _3nd_group = ['restecg']
            _4th_group = ['slope']
            df.loc[:, tuple(_1st_group)] = self.estimate_by_mice(df[_1st_group], 1, 1)
            df.loc[:, tuple(_3nd_group)] = self.estimate_by_mice(df[_3nd_group], 1, 3)
            df.loc[:, tuple(_4th_group)] = self.estimate_by_mice(df[_4th_group], 1, 4)

        return df


    def categorize_numerical_values(self,df):
        self.categorize_age(df)
        df['age'] = df['Age_band']

        self.categorize_chol(df)
        df['chol'] = df['chol_band']

        self.categorize_trestbps(df)
        df['trestbps'] = df['trestbps_band']

        self.categorize_oldpeak(df)
        df['oldpeak'] = df['oldpeak_band']

        self.categorize_thalach(df)
        df['thalach'] = df['thalach_band']
        df.drop(['Age_band', 'chol_band', 'trestbps_band', 'oldpeak_band', 'thalach_band'], axis=1, inplace=True)

    @staticmethod
    def categorize_age(df):
        df['Age_band'] = 0
        df.loc[(df['age'] >= 28) & (df['age'] <= 45), 'Age_band'] = 0
        df.loc[(df['age'] > 45) & (df['age'] <= 52), 'Age_band'] = 1
        df.loc[(df['age'] > 52) & (df['age'] <= 57), 'Age_band'] = 2
        df.loc[(df['age'] > 57) & (df['age'] <= 62), 'Age_band'] = 3
        df.loc[df['age'] > 62, 'Age_band'] = 4

    @staticmethod
    def categorize_chol(df):
        df['chol_band'] = 0
        df.loc[(df['chol'] >= 85) & (df['chol'] <= 211), 'chol_band'] = 0
        df.loc[(df['chol'] > 211) & (df['chol'] <= 238), 'chol_band'] = 1
        df.loc[(df['chol'] > 238) & (df['chol'] <= 252), 'chol_band'] = 2
        df.loc[(df['chol'] > 252) & (df['chol'] <= 275), 'chol_band'] = 3
        df.loc[df['chol'] > 275, 'chol_band'] = 4

    @staticmethod
    def categorize_trestbps(df):
        df['trestbps_band'] = 0
        df.loc[(df['trestbps'] <= 120), 'trestbps_band'] = 0
        df.loc[(df['trestbps'] > 120) & (df['trestbps'] <= 128), 'trestbps_band'] = 1
        df.loc[(df['trestbps'] > 128) & (df['trestbps'] <= 135), 'trestbps_band'] = 2
        df.loc[(df['trestbps'] > 135) & (df['trestbps'] <= 145), 'trestbps_band'] = 3
        df.loc[df['trestbps'] > 145, 'trestbps_band'] = 4

    @staticmethod
    def categorize_thalach(df):
        df['thalach_band'] = 0
        df.loc[(df['thalach'] >= 60) & (df['thalach'] <= 116), 'thalach_band'] = 0
        df.loc[(df['thalach'] > 116) & (df['thalach'] <= 131), 'thalach_band'] = 1
        df.loc[(df['thalach'] > 131) & (df['thalach'] <= 144), 'thalach_band'] = 2
        df.loc[(df['thalach'] > 144) & (df['thalach'] <= 160), 'thalach_band'] = 3
        df.loc[df['thalach'] > 160, 'thalach_band'] = 4

    @staticmethod
    def categorize_oldpeak(df):
        df['oldpeak'] = df['oldpeak'].astype('float32')
        df['oldpeak_band'] = 0
        df.loc[(df['oldpeak'] >= -3) & (df['oldpeak'] <= -0.8), 'oldpeak_band'] = 0
        df.loc[(df['oldpeak'] > -0.8) & (df['oldpeak'] <= 0.111), 'oldpeak_band'] = 1
        df.loc[(df['oldpeak'] > 0.111) & (df['oldpeak'] <= 0.845), 'oldpeak_band'] = 2
        df.loc[(df['oldpeak'] > 0.45), 'oldpeak_band'] = 3



    def feature_engineering(self, df):
        # df['LotAreaSquareMeters'] = self.square_feet_to_meters(df.LotArea.values)
        self.skew_correction(df)

    @staticmethod
    def skew_correction(df):
        # Skew correction
        # compute skewness
        skewed_feats = df[df.columns[df.columns != 'fbs']].apply(lambda x: skew(x.dropna()))
        skewed_feats = skewed_feats[skewed_feats > 0.75]
        skewed_feats = skewed_feats.index
        df.loc[:, tuple(skewed_feats)] = np.log1p(np.asarray(df[skewed_feats], dtype=float))
        # df[skewed_feats] = np.log1p(np.asarray(df[skewed_feats], dtype=float))

    @staticmethod
    def outlier_prediction(x_train, y_train):
        # Use built-in isolation forest or use predicted vs. actual
        # Compute squared residuals of every point
        # Make a threshold criteria for inclusion

        # The prediction returns 1 if sample point is inlier. If outlier prediction returns -1
        rng = np.random.RandomState(42)
        clf_all_features = IsolationForest(max_samples=100, random_state=rng)
        clf_all_features.fit(x_train)

        # Predict if a particular sample is an outlier using all features for higher dimensional
        # data set.
        y_pred_train = clf_all_features.predict(x_train)

        # Exclude suggested outlier samples for improvement of prediction power/score
        outlier_map_out_train = np.array(map(lambda x: x == 1, y_pred_train))
        x_train_modified = x_train[outlier_map_out_train,]
        y_train_modified = y_train[outlier_map_out_train,]

        return x_train_modified, y_train_modified

    def drop_feature_before_preparation(self, df):
        # Acceptable limit of NaN in features
        limit_of_nans = 0.5 * df.shape[0]
        # limit_of_nans = 800
        for feature in self.features_with_missing_values_in_dataframe(df).index:
            if df[feature].isnull().sum() > limit_of_nans:
                df = df.drop([feature], axis=1)
        return df

    def drop_variable(self, df):
        self.df_test_all_feature_var_names = df.columns
        return df



    @staticmethod
    def one_hot_encoder_all(df):
        df_class = df.copy()
        ohe = OneHotEncoder()
        for feature, value in df_class.items():
            df_class[feature] = df_class[feature].astype(int)
            if feature == 'age':
                array_X = df_class['age'].values.reshape(df_class['age'].shape[0], -1)
                label_classes = df_class['age'].unique()
                label_classes = np.sort(label_classes)
                new_one_hot_encoded_features = [''.join([feature, '_', str(x)]) for x in label_classes]
                feature_var_values = ohe.fit_transform(array_X).toarray()
                df_age = pd.DataFrame(feature_var_values, columns=new_one_hot_encoded_features)
            elif feature == 'sex':
                array_X = df_class['sex'].reshape(df_class['sex'].shape[0], -1)
                label_classes = df_class['sex'].unique()
                label_classes = np.sort(label_classes)
                new_one_hot_encoded_features = [''.join([feature, '_', str(x)]) for x in label_classes]
                feature_var_values = ohe.fit_transform(array_X).toarray()
                df_sex = pd.DataFrame(feature_var_values, columns=new_one_hot_encoded_features)
            elif feature == 'cp':
                array_X = df_class['cp'].values.reshape(df['cp'].shape[0], -1)

                label_classes = df_class['cp'].unique()
                label_classes = np.sort(label_classes)
                new_one_hot_encoded_features = [''.join([feature, '_', str(x)]) for x in label_classes]
                feature_var_values = ohe.fit_transform(array_X).toarray()
                df_cp = pd.DataFrame(feature_var_values, columns=new_one_hot_encoded_features)
            elif feature == 'trestbps':
                array_X = df_class['trestbps'].reshape(df_class['trestbps'].shape[0], -1)
                label_classes = df_class['trestbps'].unique()
                label_classes = np.sort(label_classes)
                new_one_hot_encoded_features = [''.join([feature, '_', str(x)]) for x in label_classes]
                feature_var_values = ohe.fit_transform(array_X).toarray()
                df_trestbps = pd.DataFrame(feature_var_values, columns=new_one_hot_encoded_features)
            elif feature == 'chol':
                array_X = df_class['chol'].reshape(df_class['chol'].shape[0], -1)
                label_classes = df_class['chol'].unique()
                label_classes = np.sort(label_classes)
                new_one_hot_encoded_features = [''.join([feature, '_', str(x)]) for x in label_classes]
                feature_var_values = ohe.fit_transform(array_X).toarray()
                df_chol = pd.DataFrame(feature_var_values, columns=new_one_hot_encoded_features)
            elif feature == 'fbs':
                array_X = df_class['fbs'].reshape(df_class['fbs'].shape[0], -1)
                label_classes = df_class['fbs'].unique()
                label_classes = np.sort(label_classes)
                new_one_hot_encoded_features = [''.join([feature, '_', str(x)]) for x in label_classes]
                feature_var_values = ohe.fit_transform(array_X).toarray()
                df_fbs = pd.DataFrame(feature_var_values, columns=new_one_hot_encoded_features)
            elif feature == 'restecg':
                array_X = df_class['restecg'].reshape(df_class['restecg'].shape[0], -1)
                label_classes = df_class['restecg'].unique()
                label_classes = np.sort(label_classes)
                new_one_hot_encoded_features = [''.join([feature, '_', str(x)]) for x in label_classes]
                feature_var_values = ohe.fit_transform(array_X).toarray()
                df_restecg = pd.DataFrame(feature_var_values, columns=new_one_hot_encoded_features)
            elif feature == 'thalach':
                array_X = df_class['thalach'].reshape(df_class['thalach'].shape[0], -1)
                label_classes = df_class['thalach'].unique()
                label_classes = np.sort(label_classes)
                new_one_hot_encoded_features = [''.join([feature, '_', str(x)]) for x in label_classes]
                feature_var_values = ohe.fit_transform(array_X).toarray()
                df_thalach = pd.DataFrame(feature_var_values, columns=new_one_hot_encoded_features)
            elif feature == 'exang':
                array_X = df_class['exang'].reshape(df_class['exang'].shape[0], -1)
                label_classes = df_class['exang'].unique()
                label_classes = np.sort(label_classes)
                new_one_hot_encoded_features = [''.join([feature, '_', str(x)]) for x in label_classes]
                feature_var_values = ohe.fit_transform(array_X).toarray()
                df_exang = pd.DataFrame(feature_var_values, columns=new_one_hot_encoded_features)
            elif feature == 'oldpeak':
                array_X = df_class['oldpeak'].reshape(df_class['oldpeak'].shape[0], -1)
                label_classes = df_class['oldpeak'].unique()
                label_classes = np.sort(label_classes)
                new_one_hot_encoded_features = [''.join([feature, '_', str(x)]) for x in label_classes]
                feature_var_values = ohe.fit_transform(array_X).toarray()
                df_oldpeak = pd.DataFrame(feature_var_values, columns=new_one_hot_encoded_features)
            elif feature == 'slope':
                array_X = df_class['slope'].reshape(df_class['slope'].shape[0], -1)
                label_classes = df_class['slope'].unique()
                label_classes = np.sort(label_classes)
                new_one_hot_encoded_features = [''.join([feature, '_', str(x)]) for x in label_classes]
                feature_var_values = ohe.fit_transform(array_X).toarray()
                df_slope = pd.DataFrame(feature_var_values, columns=new_one_hot_encoded_features)
                '''
            elif feature == 'heartdisease':
                array_X = df_class['heartdisease'].reshape(df_class['heartdisease'].shape[0], -1)
                label_classes = df_class['heartdisease'].unique()
                label_classes = np.sort(label_classes)
                new_one_hot_encoded_features = [''.join([feature, '_', str(x)]) for x in label_classes]
                feature_var_values = ohe.fit_transform(array_X).toarray()
                df_heartdisease = pd.DataFrame(feature_var_values, columns=new_one_hot_encoded_features)
            '''
            elif feature == 'heartdisease':
                #array_X = np.asarray(df_class['heartdisease'].values)
                df_heartdisease = pd.DataFrame(data=df_class['heartdisease'].values, columns=['heartdisease'])
                #df_heartdisease = df_class['heartdisease'];#pd.DataFrame(array_X, columns='heartdisease')
        df = pd.concat([df_age, df_sex, df_cp, df_trestbps, df_chol, df_fbs,df_restecg, df_thalach, df_exang, df_oldpeak, df_slope, df_heartdisease ], axis=1)
        return df

    @staticmethod
    def one_hot_encoder_categorical(df):
        df_class = df.copy()
        ohe = OneHotEncoder()
        for feature, value in df_class.items():

            if feature == 'age':
               df_age = pd.DataFrame(df_class['age'].values, columns=['age'])
            elif feature == 'sex':
                df_class[feature] = df_class[feature].astype(int)
                array_X = df_class['sex'].values.reshape(df_class['sex'].shape[0], -1)
                label_classes = df_class['sex'].unique()
                label_classes = np.sort(label_classes)
                new_one_hot_encoded_features = [''.join([feature, '_', str(x)]) for x in label_classes]
                feature_var_values = ohe.fit_transform(array_X).toarray()
                df_sex = pd.DataFrame(feature_var_values, columns=new_one_hot_encoded_features)
            elif feature == 'cp':
                df_class[feature] = df_class[feature].astype(int)
                array_X = df_class['cp'].values.reshape(df['cp'].shape[0], -1)
                label_classes = df_class['cp'].unique()
                label_classes = np.sort(label_classes)
                new_one_hot_encoded_features = [''.join([feature, '_', str(x)]) for x in label_classes]
                feature_var_values = ohe.fit_transform(array_X).toarray()
                df_cp = pd.DataFrame(feature_var_values, columns=new_one_hot_encoded_features)
            elif feature == 'trestbps':
                df_trestbps = pd.DataFrame(df_class['trestbps'].values, columns=['trestbps'])
            elif feature == 'chol':
                df_chol = pd.DataFrame(df_class['chol'].values, columns=['chol'])
            elif feature == 'fbs':
                df_class[feature] = df_class[feature].astype(int)
                array_X = df_class['fbs'].values.reshape(df_class['fbs'].shape[0], -1)
                label_classes = df_class['fbs'].unique()
                label_classes = np.sort(label_classes)
                new_one_hot_encoded_features = [''.join([feature, '_', str(x)]) for x in label_classes]
                feature_var_values = ohe.fit_transform(array_X).toarray()
                df_fbs = pd.DataFrame(feature_var_values, columns=new_one_hot_encoded_features)
            elif feature == 'restecg':
                df_restecg = pd.DataFrame(df_class['restecg'].values, columns=['restecg'])
            elif feature == 'thalach':
                df_thalach = pd.DataFrame(df_class['thalach'].values, columns=['thalach'])
            elif feature == 'exang':
                df_class[feature] = df_class[feature].astype(int)
                array_X = df_class['exang'].values.reshape(df_class['exang'].shape[0], -1)
                label_classes = df_class['exang'].unique()
                label_classes = np.sort(label_classes)
                new_one_hot_encoded_features = [''.join([feature, '_', str(x)]) for x in label_classes]
                feature_var_values = ohe.fit_transform(array_X).toarray()
                df_exang = pd.DataFrame(feature_var_values, columns=new_one_hot_encoded_features)
            elif feature == 'oldpeak':
                df_oldpeak = pd.DataFrame(df_class['oldpeak'].values, columns=['oldpeak'])
            elif feature == 'slope':
                df_class[feature] = df_class[feature].astype(int)
                array_X = df_class['slope'].values.reshape(df_class['slope'].shape[0], -1)
                label_classes = df_class['slope'].unique()
                label_classes = np.sort(label_classes)
                new_one_hot_encoded_features = [''.join([feature, '_', str(x)]) for x in label_classes]
                feature_var_values = ohe.fit_transform(array_X).toarray()
                df_slope = pd.DataFrame(feature_var_values, columns=new_one_hot_encoded_features)
            elif feature == 'heartdisease':
                df_class[feature] = df_class[feature].astype(int)
                df_heartdisease = pd.DataFrame(data=df_class['heartdisease'].values, columns=['heartdisease'])

        df = pd.concat(
            [df_age, df_sex, df_cp, df_trestbps, df_chol, df_fbs, df_restecg, df_thalach, df_exang, df_oldpeak,
             df_slope, df_heartdisease], axis=1)
        return df

    def feature_scaling(self, df):
        df = df.copy()
        # Standardization (centering and scaling) of dataset that removes mean and scales to unit variance
        standard_scaler = StandardScaler()
        relevant_features = ClinicalData._numerical_feature_names
        res = standard_scaler.fit_transform(X=df[relevant_features].values)
        df.loc[:, tuple(relevant_features)] = res
        return df


    def prepare_data(self, df):
        df = df.copy()
        # drop features with > 50% Nans
        df = self.drop_feature_before_preparation(df)
        #self.feature_engineering(df)
        df = self.standardize_oldpeak(df) #since it contains -neg values
        # Todo: clean_data -> imputation
        df = self.clean_data(df)
        #normalizing_data_set
      #  df = self.feature_scaling(df)
        # Todo: OHE -> categorical variables
      #  df = self.one_hot_encoder_categorical(df)
        #df = self.one_hot_encoder_all(df)

        return df




    @staticmethod
    def estimate_by_mice(df, _iscategorical, group):
        df_estimated_var = df.copy()
        random.seed(129)
        mice = MICE()  # model=RandomForestClassifier(n_estimators=100))
        array_X = np.asarray(df.values, dtype=float)
        if array_X.ndim < 2:
            array_X = array_X.reshape(array_X.shape[0],-1)
            res = mice.complete(array_X, _iscategorical, group)
            df_estimated_var.loc[:, :] = res[:][:]
        else:
            res = mice.complete(array_X,_iscategorical, group)
        if group == 3:
            df_estimated_var['restecg'] = res[:][:]
        elif group == 4:
            df_estimated_var['slope'] = res[:][:]
        else:
            df_estimated_var.loc[:, df.columns] = res[:][:]
        return df_estimated_var

    def df_missing_to_nan(self,df):
        df_with_nan = df.copy()

        for feature in df_with_nan.columns:
            df_with_nan[feature] = df_with_nan[feature].replace([-np.inf, np.inf, -9], np.nan)
        return df_with_nan



    def standardize_oldpeak(self,df):
        feature = 'oldpeak'
        standard_scaler = StandardScaler()
        df[feature] = df[feature].replace([-np.inf, np.inf, -9], np.nan)
        mask = ~df[feature].isnull()
        array_oldpeak_values = np.asarray(df[feature][mask].values).reshape(-1, 1)
        res = standard_scaler.fit_transform(array_oldpeak_values)

        if mask.sum().sum() > 0:
            mask = ~df[feature].isnull()
            mask_index = mask[mask == 1].index
            df.loc[mask_index, tuple([feature])] = res[:, :]
        else:
            df.loc[:, tuple(feature)] = res
        return df






    @staticmethod
    def standardize_relevant_features(df, relevant_features, res):
        i_column = 0
        for feature in relevant_features:
            mask = ~df[feature].isnull()
            mask_index = mask[mask == 1].index
            df.loc[mask_index, tuple([feature])] = res[:, i_column]
            i_column += 1
        return df

    def missing_values_in_dataframe(self, df):
        mask = self.features_with_null_logical(df)
        print(df[mask[mask == 0].index.values].isnull().sum())
        print('\n')

    def features_with_missing_values_in_dataframe(self, df):
        df = df.copy()
        mask = self.features_with_null_logical(df)
        return df[mask[mask == 0].index.values].isnull().sum()

    @staticmethod
    def rmse_cv(model, x_train, y_train):
        rmse = np.sqrt(-cross_val_score(model, x_train, y_train,
                                        scoring='neg_mean_squared_error', cv=5))
        return rmse

    @staticmethod
    def features_with_null_logical(df, axis=1):
        row_length = len(df._get_axis(0))
        # Axis to count non null values in. aggregate_axis=0 implies counting for every feature
        aggregate_axis = 1 - axis
        features_non_null_series = df.count(axis=aggregate_axis)
        # Whenever count() differs from row_length it implies a null value exists in feature
        # column and a False in mask
        mask = row_length == features_non_null_series
        return mask


    @staticmethod
    def rmse(y_pred, y_actual):
        n_samples = np.shape(y_pred)[0]
        squared_residuals_summed = 0.5 * sum((y_pred - y_actual) ** 2)
        return np.sqrt(2.0 * squared_residuals_summed / n_samples)

    def outlier_identification(self, model, x_train, y_train):
        # Split the training data into an extra set of test
        x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(x_train,
                                                                                    y_train)
        print('\nOutlier shapes')
        print(np.shape(x_train_split), np.shape(x_test_split), np.shape(y_train_split),
              np.shape(y_test_split))
        model.fit(x_train_split, y_train_split)
        y_predicted = model.predict(x_test_split)
        residuals = np.absolute(y_predicted - y_test_split)
        rmse_pred_vs_actual = self.rmse(y_predicted, y_test_split)
        outliers_mask = residuals >= rmse_pred_vs_actual
        # outliers_mask = np.insert(np.zeros((np.shape(y_train_split)[0],), dtype=np.int),
        # np.shape(y_train_split)[0], outliers_mask)
        outliers_mask = np.concatenate([np.zeros((np.shape(y_train_split)[0],), dtype=bool),
                                        outliers_mask])
        not_an_outlier = outliers_mask == 0
        # Resample the training set from split, since the set was randomly split
        x_out = np.insert(x_train_split, np.shape(x_train_split)[0], x_test_split, axis=0)
        y_out = np.insert(y_train_split, np.shape(y_train_split)[0], y_test_split, axis=0)
        return x_out[not_an_outlier,], y_out[not_an_outlier,]

    def drop_variable_before_preparation(self, df):
        # Acceptable limit of NaN in features
        limit_of_nans = 0.45* df.shape[0]
        # limit_of_nans = 800
        for feature in self.features_with_missing_values_in_dataframe(df).index:
            if df[feature].isnull().sum() > limit_of_nans:
                df = df.drop([feature], axis=1)

        return df




if __name__ == '__main__':
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import SGDRegressor
    from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
    # from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import SelectFromModel
    # from sklearn.naive_bayes import GaussianNB
    # from sklearn import svm
    # from collections import OrderedDict
    # from sklearn.ensemble import IsolationForest
    import seaborn as sns
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import GridSearchCV
    # from sklearn.model_selection import KFold, train_test_split


    ''' Prepare data '''

    clinical_data = ClinicalData()
    df = clinical_data.df.copy()
    df_with_nan = clinical_data.df_missing_to_nan(df)
    df_prepared = clinical_data.prepare_data(df_with_nan)
    print(pd.__version__)
    filename = 'df_imputed_vclean.pkl'
    df_prepared.to_pickle(filename)

    mydf = pd.read_pickle(filename)
    mydf.info()



'''
    df = clinical_data.prepare_data_random_forest(df)

    print('\n TRAINING DATA:----------------------------------------------- \n')
    print(df.head(3))
    print('\n')
    print(df.info())
    print('\n')
    print(df.describe())
'''