import pandas as pd
import numpy as np
import pywt
import math
from sklearn.model_selection import StratifiedShuffleSplit
# import matplotlib.pyplot as plt
# %matplotlib inline
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from joblib import dump
from sklearn.preprocessing import OrdinalEncoder
from scipy import signal
from sklearn.preprocessing import StandardScaler


def read_data(file):
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # raw_data = pd.read_excel(file)
    raw_data = pd.read_csv(file)
    raw_data.columns = raw_data.columns.str.strip().str.lower()
    last_column = raw_data.columns[-1]
    to_use_in_pat =f"^chanel|^channel|{last_column}"
    # raw_data = pd.read_feather(file)
    raw_data = raw_data.loc[:, raw_data.columns.str.contains(pat=to_use_in_pat)].copy()
    #breakpoint()

    #if "Unnamed: 0" in raw_data.columns:
     #   raw_data = raw_data.drop(labels=["Unnamed: 0", "time"], axis="columns")
    #else:
       # raw_data = raw_data.drop(labels=["time"], axis="columns")


    # https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
    # raw_data.iloc[:,0:-1] = scaler.fit_transform(raw_data.iloc[:,0:-1].to_numpy())
    # select_channels = raw_data.iloc[:,:8]
    # max_value = select_channels.max()
    # min_value = select_channels.min()

    # raw_data.sort_values("position")
    # raw_data.iloc[:, :8] = raw_data.iloc[:, :8] * 1000
    return raw_data


def spare_classes_(data):
    positions = data.groupby(data.columns.values[-1])
    classes = [positions.get_group(i + 1) for i, j in enumerate(positions)]
    return classes


def standardize_classes(classes, time_between_captures_of_samples, window_time):
    n_of_samples = window_time // time_between_captures_of_samples
    for i, j in enumerate(classes):
        lenght_ = int((len(j) // n_of_samples) * n_of_samples)
        #classes[i] = classes[i].iloc[:lenght_, :8]
        classes[i] = classes[i].iloc[:lenght_, :-1]
    return classes


def to_numpy_func(classes):
    for i, j in enumerate(classes):
        classes[i] = j.to_numpy()
    classes = np.array(classes, dtype=object)

    return classes


def sample_classes_(classes, time_between_captures_of_samples, window_time, n_of_channels_and_category):
    n_of_classes = len(classes)
    n_samples = int(window_time / time_between_captures_of_samples)
    class_mod_f = []
    for i, j in enumerate(classes):
        len_ = int(len(j) / n_samples)
        class_mod_ = np.zeros([len_, n_samples, n_of_channels_and_category])
        for k in range(len_):
            for l in range(n_samples):
                class_mod_[k, l, :] = classes[i][l + k * n_samples][:]
        class_mod_f.append(class_mod_)
    class_mod_f = np.array(class_mod_f, dtype=object)

    return class_mod_f


def filter_iir(sig, sos):
    sig = signal.sosfilt(sos, sig)
    return sig


def filter_signal(class_mod_, sos, n_of_channels_and_category):
    for i, j in enumerate(class_mod_):
        # l_class_ = len(class_mod_[i])
        for k, l in enumerate(class_mod_[i]):
            for m in range(n_of_channels_and_category):
                # signal_mod = class_mod_[i][k, :, l]
                # signal_mod = filter_iir(sig=signal_mod, sos=sos)
                class_mod_[i][k, :, m] = signal.sosfilt(sos, class_mod_[i][k, :, m])

                # class_mod_[i][k, :, l] = signal_mod
    return class_mod_


def wav_filter(signal, filter_to_use, levels_to_use, layers_to_catch):
    a = 1
    Coeffs = pywt.wavedec(signal, filter_to_use, level=levels_to_use)

    for i in range(1, -1, -(levels_to_use + 1)):
        if -i not in layers_to_catch:
            Coeffs[i] = np.zeros_like(Coeffs[i])
    Rec = pywt.waverec(Coeffs, filter_to_use)
    return Rec


def select_wavelet_layer_x(class_mod_, filter_to_use, levels_to_use,
                           layers_to_catch, n_of_channels_and_category):
    for i, j in enumerate(class_mod_):
        # l_class_ = int(len(class_mod_[i]))
        for k, l in enumerate(class_mod_[i]):
            for m in range(n_of_channels_and_category):
                # signal_mod = class_mod_[i][k, :, m]
                class_mod_[i][k, :, m] = wav_filter(signal=class_mod_[i][k, :, m],
                                                    filter_to_use=filter_to_use,
                                                    levels_to_use=levels_to_use,
                                                    layers_to_catch=layers_to_catch
                                                    )
                # class_mod_[i][k, :, l] = signal_mod
    return class_mod_


def m_mav_values_(class_mod_, time_between_captures_of_samples, window_time, n_of_channels_and_category):
    mav_table_ = []
    n_of_samples = int(window_time / time_between_captures_of_samples)

    for i, j in enumerate(class_mod_):
        acum_x = 0
        l_class_ = int(len(j))
        s = [l_class_, n_of_channels_and_category]
        m_class_ = np.zeros(s)
        for k in range(l_class_):
            for l in range(n_of_channels_and_category):
                for m in range(n_of_samples):
                    acum_x += abs(j[k, m, l]) / n_of_samples
                m_class_[k, l] = acum_x
                acum_x = 0
        mav_table_.append(m_class_)
    mav_table_ = np.array(mav_table_, dtype=object)
    return mav_table_


def m_rms_values_(class_mod_, time_between_captures_of_samples, window_time, n_of_channels_and_category):
    rms_table_ = []
    n_of_samples = int(window_time / time_between_captures_of_samples)

    for i, j in enumerate(class_mod_):
        acum_x = 0
        l_class_ = int(len(j))
        s = [l_class_, n_of_channels_and_category]
        m_class_ = np.zeros(s)
        for k in range(l_class_):
            for l in range(n_of_channels_and_category):
                for m in range(n_of_samples):
                    acum_x += ((j[k, m, l]) ** 2) / n_of_samples
                m_class_[k, l] = math.sqrt(acum_x)
                acum_x = 0
        rms_table_.append(m_class_)
    rms_table_ = np.array(rms_table_, dtype=object)
    return rms_table_


def matrix_m(type_matrix, class_mod_, time_between_captures_of_samples, window_time, n_of_channels_and_category):
    if type_matrix == "rms":
        m_matrix_ = m_rms_values_(class_mod_, time_between_captures_of_samples, window_time, n_of_channels_and_category)
    elif type_matrix == "mav":
        m_matrix_ = m_mav_values_(class_mod_, time_between_captures_of_samples, window_time, n_of_channels_and_category)
    return m_matrix_


def fitting_m_class(m_matrix_):
    m_matrix_stack = None
    lens_of_classes = []
    for i, j in enumerate(m_matrix_):
        if i == 0:
            m_matrix_stack = j
        else:
            m_matrix_stack = np.vstack((m_matrix_stack, j))
        lens_of_classes.append(len(j))

    m_matrix_table_ = np.array(m_matrix_stack, dtype=object)
    lens_table_ = np.array(lens_of_classes, dtype=object)
    return m_matrix_table_, lens_table_


def name_columns_of_table(n_columns):
    columns_name = None
    for i in range(n_columns):
        if columns_name is None:
            columns_name = [f"Chanel_{i+1}"]
        else:
            if i < n_columns - 1:
                columns_name.append(f"Chanel_{i+1}")
            else:
                columns_name.append("Category")
    return columns_name


def transf_to_df_class(m_matrix_, lens_table_):
    #from sklearn.preprocessing import StandardScaler
    #scaler = StandardScaler()
    # https://stackoverflow.com/questions/8486294/how-to-add-an-extra-column-to-a-numpy-array
    m_matrix_ = np.hstack((m_matrix_, np.zeros((m_matrix_.shape[0], 1))))
    lenght_ = 0
    for i, j in enumerate(lens_table_):
        for k in range(int(j)):
            m_matrix_[lenght_ + k, -1] = i
        lenght_ += int(j)
    n_columns = len(m_matrix_[0])
    name_of_columns = name_columns_of_table(n_columns=n_columns)
    #print(name_of_columns)
    #df = pd.DataFrame(m_matrix_, columns=['Chanel_1',
     #                                     'Chanel_2',
      #                                    'Chanel_3',
       #                                   'Chanel_4',
        #                                  'Chanel_5',
         #                                 'Chanel_6',
          #                                'Chanel_7',
           #                               'Chanel_8',
            #                              'Category'
             #                             ])

    df = pd.DataFrame(m_matrix_, columns=name_of_columns)
    # https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
    # 3df.iloc[:,0:-1] = scaler.fit_transform(df.iloc[:,0:-1].to_numpy())
    return df


def Strat_train_test(dataframe, test_size, random_state):
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in split.split(dataframe, dataframe['Category']):
        train = dataframe.loc[train_index]
        test = dataframe.loc[test_index]
    return train, test


def sklearn_spare_test_train(dataframe, test_size, random_state):
    Train, Test = Strat_train_test(dataframe=dataframe, test_size=test_size, random_state=random_state)
    m_class_Train_IndepentVars = Train.iloc[:, :8]
    m_class_Train_TargetVar = Train.loc[:, "Category"]
    m_class_Test_IndepentVars = Test.iloc[:, :8]
    m_class_Test_TargetVar = Test.loc[:, "Category"]
    return m_class_Train_IndepentVars, m_class_Train_TargetVar, m_class_Test_IndepentVars, m_class_Test_TargetVar


def decomposition_PCA(m_class_Train_IndepentVar, var):
    pca = PCA(var)
    pca.fit(m_class_Train_IndepentVar)
    m_class_Train_IndepentVar_PCA = pca.transform(m_class_Train_IndepentVar)
    return m_class_Train_IndepentVar_PCA


def encode_data(m_class_Train_TargetVar, m_class_Test_TargetVar):
    ordinal_encoder = OrdinalEncoder()
    m_class_Train_TargetVar = ordinal_encoder.fit_transform(m_class_Train_TargetVar.reshape(-1, 1))
    m_class_Test_TargetVar = ordinal_encoder.fit_transform(m_class_Test_TargetVar.reshape(-1, 1))
    return m_class_Train_TargetVar, m_class_Test_TargetVar


def m_class_to_numpy_(m_class_Train_IndepentVars, m_class_Train_TargetVar, m_class_Test_IndepentVars,
                      m_class_Test_TargetVar):
    m_class_Train_IndepentVars = m_class_Train_IndepentVars.to_numpy()
    m_class_Train_TargetVar = m_class_Train_TargetVar.to_numpy()
    m_class_Test_IndepentVars = m_class_Test_IndepentVars.to_numpy()
    m_class_Test_TargetVar = m_class_Test_TargetVar.to_numpy()
    return m_class_Train_IndepentVars, m_class_Train_TargetVar, m_class_Test_IndepentVars, m_class_Test_TargetVar


def apl_error(model, independent_vars, target_vars, cv):
    acerto = cross_val_score(model, independent_vars, target_vars, cv=cv)
    erro = 1 - acerto.mean()
    return erro


def apply_classifiers(var_train, var_target, var_test, var_test_target, cv):
    lda = LinearDiscriminantAnalysis()
    tree = DecisionTreeClassifier()
    gnb = GaussianNB()
    lin_svm = svm.SVC(decision_function_shape='ovo')
    knn = KNeighborsClassifier(n_neighbors=5)
    lda.fit(var_train, var_target)
    tree.fit(var_train, var_target)
    gnb.fit(var_train, var_target)
    lin_svm.fit(var_train, var_target)
    knn.fit(var_train, var_target)
    classifiers = [lda, gnb, lin_svm, knn, tree]
    #error_tree_t = apl_error(tree, var_train, var_target, cv)
    #error_lda_t = apl_error(lda, var_train, var_target, cv)
    #error_gnb_t = apl_error(gnb, var_train, var_target, cv)
    #error_lin_svm_t = apl_error(lin_svm, var_train, var_target, cv)
    #error_neigh_t = apl_error(neigh, var_train, var_target, cv)
    dump_classifiers(lda, tree, gnb, lin_svm, knn)
    #pred_using_knn = lda.predict(var_test)
    #print(accuracy_score(var_test_target, pred_using_knn))
    store_accuracy(classifiers, var_test, var_test_target)
    #return error_tree_t, error_lda_t, error_gnb_t, error_lin_svm_t, error_neigh_t
    return lda, knn, gnb, tree, lin_svm


def store_accuracy(classifiers, var_test, var_test_target):
    names = ["lda", "gnb", "lin_svm", "knn", "tree"]
    accuracy = []
    for i, j in enumerate(classifiers):
        pred = j.predict(var_test)
        score = accuracy_score(var_test_target, pred)
        accuracy.append(score)
    d = {'Classifiers': names, 'Scores': accuracy}
    df = pd.DataFrame(data=d)
    df.to_csv("scores_of_classifiers.csv", index=False)

def dump_classifiers(lda, tree, gnb, lin_svm, knn):
    dump(lda, 'files_joblib/lda_teste.joblib')
    dump(tree, 'files_joblib/tree_teste.joblib')
    dump(gnb, 'files_joblib/gnb_teste.joblib')
    dump(lin_svm, 'files_joblib/lin_svm_teste.joblib')
    dump(knn, 'files_joblib/neigh_teste.joblib')
    # copy_to_my_arm_def the .jobfiles files
    dump(lda, '../my_arm_def/lda_teste.joblib')
    dump(tree, '../my_arm_def/tree_teste.joblib')
    dump(gnb, '../my_arm_def/gnb_teste.joblib')
    dump(lin_svm, '../my_arm_def/lin_svm_teste.joblib')
    dump(knn, '../my_arm_def/neigh_teste.joblib')


#def predict_data(data_):
    #from joblib import load
    #clf = load('/home/caio/ros2_ws/lda.joblib')
    #p = clf.predict(data_)
    #return p[0]
