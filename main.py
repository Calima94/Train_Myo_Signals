from train_emg_data import Ui_Form
from PyQt5 import QtCore, QtGui, QtWidgets
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
import sys
from matplotlib.figure import Figure
from numpy import random
import pandas as pd
import numpy as np
import train_signals_emg
import os


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=4, height=5, dpi=200):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(100)
        super(MplCanvas, self).__init__(fig)


class EmgApplication(Ui_Form):
    def __init__(self, definitions):
        super(Ui_Form, self).__init__()
        self.data_and_classifiers = None
        self.setupUi(definitions)
        self.trained_sys = False

        self.search_file_btn.clicked.connect(self.find_path_file)
        self.filter_1_btn.clicked.connect(self.find_path_filter_1)
        self.filter_2_btn.clicked.connect(self.find_path_filter_2)
        self.redefine_param_btn.clicked.connect(self.refresh_params)

        self.plot_conf_matrix_btn_2.clicked.connect(self.plot_confusion_matrix)


        # Select pages
        self.results_btn.clicked.connect(lambda: self.pages.setCurrentWidget(self.results_pg))
        self.home_btn.clicked.connect(lambda: self.pages.setCurrentWidget(self.home_pg))
        self.def_btn.clicked.connect(lambda: self.pages.setCurrentWidget(self.def_pg))
        self.contact_btn.clicked.connect(lambda: self.pages.setCurrentWidget(self.contact_pg))

        # Main button

        self.train_all_signals.clicked.connect(self.train_myo)

        self.plot_process_signals_btn.clicked.connect(
            lambda: self.results_pgs.setCurrentWidget(self.plot_processed_signals_pg))
        self.plot_raw_signals_btn.clicked.connect(
            lambda: self.results_pgs.setCurrentWidget(self.plot_raw_signals_pg))

        self.set_of_canvas()
        self.btn_plot_processed_signals.clicked.connect(self.classified_signals)
        self.plot_raw_signals_btn_2.clicked.connect(self.raw_signals_plot)

    def plot_confusion_matrix(self):
        if self.data_and_classifiers is not None:

            self.figure_2.clear()
            # pred_knn = self.data_and_classifiers.knn.predict(self.data_and_classifiers.m_class_test_independentVars)
            classifiers = {'knn': self.data_and_classifiers.knn,
                           'lda': self.data_and_classifiers.lda,
                           'gnb': self.data_and_classifiers.gnb,
                           'tree': self.data_and_classifiers.tree,
                           'lin_svm': self.data_and_classifiers.lin_svm}

            conf_classifier = self.confusion_matrix_classifier_cb.currentText()
            conf_type = self.confusin_matrix_type_cb.currentText()
            if conf_type == 'None':
                conf_type = None
            # https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
            ax4 = self.figure_2.add_subplot(111)
            ConfusionMatrixDisplay.from_estimator(classifiers[conf_classifier],
                                                  self.data_and_classifiers.m_class_test_independentVars,
                                                  self.data_and_classifiers.m_class_test_target_vars,
                                                  ax=ax4,
                                                  normalize=conf_type)

            ax4.set(xlabel='Predicted', ylabel='True', title='Confusion Matrix True vs Predicted')



            #ax.bar(classifiers, values)
            #ax3.set_title(f'Scores of the classifiers')
            #ax3.set_xlabel(f'Classifiers')
            #ax3.set_ylabel(f'Scores %')

            # refresh canvas
            self.canvas_conf_matrix.draw()

    def refresh_params(self):
        frequency = self.frequency.text()
        window_time = self.window_time.text()
        n_of_channels = self.n_of_channels.currentText()
        filter_1_path = self.filter_1_path.text()
        filter_2_path = self.filter_2_path.text()
        wav_filter = self.wav_filter.text()
        levels_of_filter = self.levels_of_wav_filter.currentText()
        layers_to_use_num = int(self.layers_to_use.currentText())
        layers_to_use = []
        for i in range(layers_to_use_num):
            layers_to_use.append(i+1)

        type_of_matrix = self.type_of_matrix.currentText()
        test_size = self.test_size.text()
        random_state = self.random_state.text()
        cv = self.cv.text()
        raw_file_data = self.file_path_label.text()

        filter_1_df = pd.read_csv(filter_1_path, index_col="Filter")
        filter_2_df = pd.read_csv(filter_2_path, index_col="Filter")

        filter_1 = filter_1_df.loc["sos_high_pass_", "Value"]
        filter_2 = filter_2_df.loc["sos_bandstop_", "Value"]

        df = pd.read_csv("Parameters/parameters.csv", index_col="Parameter")

        df.loc["frequency_of_capture", "Value"] = frequency
        df.loc["window_time", "Value"] = window_time
        df.loc["n_of_channels_and_category", "Value"] = n_of_channels
        df.loc["sos_high_pass_", "Value"] = filter_2
        df.loc["sos_bandstop_", "Value"] = filter_1
        df.loc["layers_to_catch", "Value"] = layers_to_use
        df.loc["levels_to_use", "Value"] = levels_of_filter
        df.loc["filter_to_use", "Value"] = wav_filter
        df.loc["type_matrix", "Value"] = type_of_matrix
        df.loc["test_size", "Value"] = test_size
        df.loc["random_state", "Value"] = random_state
        df.loc["cv", "Value"] = cv
        df.loc["file", "Value"] = raw_file_data
        df.to_csv("Parameters/parameters.csv")


    def find_path_filter_1(self):
        # self.file_path_label.setText("This path")
        # Open a File Dialog
        if "PYCHARM_HOSTED" in os.environ:
            ret, _ = QtWidgets.QFileDialog.getOpenFileName(
                None, 'Open File',
                'Parameters/', "All Files (*);;Python Files (*.py)",
                options=QtWidgets.QFileDialog.DontUseNativeDialog
            )
        else:
            ret, _ = QtWidgets.QFileDialog.getOpenFileName(
                None, 'Open File',
                'Parameters/', "All Files (*);;Python Files (*.py)"
            )
        if ret:
            self.filter_1_path.setText(ret)

    def find_path_filter_2(self):
        # self.file_path_label.setText("This path")
        # Open a File Dialog
        if "PYCHARM_HOSTED" in os.environ:
            ret, _ = QtWidgets.QFileDialog.getOpenFileName(
                None, 'Open File',
                'Parameters/', "All Files (*);;Python Files (*.py)",
                options=QtWidgets.QFileDialog.DontUseNativeDialog
            )
        else:
            ret, _ = QtWidgets.QFileDialog.getOpenFileName(
                None, 'Open File',
                'Parameters/', "All Files (*);;Python Files (*.py)"
            )
        if ret:
            self.filter_2_path.setText(ret)


    def find_path_file(self):
        #self.file_path_label.setText("This path")
        # Open a File Dialog
        if "PYCHARM_HOSTED" in os.environ:
            ret, _ = QtWidgets.QFileDialog.getOpenFileName(
                None, 'Open File',
                'Raw_EMG_Data/', "All Files (*);;Python Files (*.py)",
                options=QtWidgets.QFileDialog.DontUseNativeDialog
            )
        else:
            ret, _ = QtWidgets.QFileDialog.getOpenFileName(
                None, 'Open File',
                'Raw_EMG_Data/', "All Files (*);;Python Files (*.py)"
            )



        #fname = QtWidgets.QFileDialog.getOpenFileName(None, "Open File", "", "All Files (*);;Python Files (*.py)")

        # Output filename to screen

        if ret:
            self.file_path_label.setText(ret)

    def train_myo(self, file):
        self.data_and_classifiers = train_signals_emg.train_signals_emg()
        self.trained_sys = True
        _translate = QtCore.QCoreApplication.translate
        df = pd.read_csv("scores_of_classifiers.csv")

        values = df["Scores"]
        #self.treeWidget.topLevelItem(0).setText(0, _translate("Form", "LDA"))
        for i, j in enumerate(values):
            value = j * 100
            self.treeWidget.topLevelItem(i).setText(1, _translate("Form", f"{value:.2f}%"))
        self.main_classifier()



    def set_of_canvas(self):

        # Main screen
        # Raw signals Canvas
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout(self.frame_15)
        self.horizontalLayout_15.setObjectName("horizontallayout_6")
        # Canvas here
        self.figure_3 = plt.figure()
        self.canvas_main_classifier = FigureCanvas(self.figure_3)
        # end of Canvas
        # Add Canvas
        self.horizontalLayout_15.addWidget(self.canvas_main_classifier)
        # end of horizontal layout



        # Raw signals Canvas
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.frame_12)
        self.horizontalLayout_5.setObjectName("horizontallayout_5")
        # Canvas here
        self.figure = plt.figure()
        self.canvas_raw_signals = FigureCanvas(self.figure)
        # end of Canvas
        # Add Canvas
        self.horizontalLayout_5.addWidget(self.canvas_raw_signals)
        # end of horizontal layout


        # Processed signals Canvas
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_13)
        self.horizontalLayout_4.setObjectName("horizontallayout_4")
        # Canvas here
        self.figure_1 = plt.figure()
        self.canvas_process_signals = FigureCanvas(self.figure_1)
        # end of Canvas
        # Add Canvas
        self.horizontalLayout_4.addWidget(self.canvas_process_signals)
        # end of horizontal layout

        # Confusion Matrix
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout(self.frame_16)
        self.horizontalLayout_16.setObjectName("horizontallayout_16")
        # Canvas here
        self.figure_2 = plt.figure()
        self.canvas_conf_matrix = FigureCanvas(self.figure_2)
        # end of Canvas
        # Add Canvas
        self.horizontalLayout_16.addWidget(self.canvas_conf_matrix)
        # end of horizontal layout


    def main_classifier(self):
        self.figure_3.clear()
        df = pd.read_csv("scores_of_classifiers.csv")
        classifiers = df["Classifiers"]
        values = df["Scores"] * 100

        ax3 = self.figure_3.add_subplot(111)
        ax3.bar(classifiers, values)
        ax3.set_title(f'Scores of the classifiers')
        ax3.set_xlabel(f'Classifiers')
        ax3.set_ylabel(f'Scores %')

        # refresh canvas
        self.canvas_main_classifier.draw()


    def raw_signals_plot(self):
        # clear the canvas
        self.figure.clear()
        channel_1 = int(self.cb_raw_signal_1.currentText())

        df = pd.read_csv("Raw_EMG_Data/train_with_openCV_list_16_05.csv")
        self.plot_values_of_chanels_raw_emg(df, channel_1)
        # refresh canvas
        self.canvas_raw_signals.draw()

    def classified_signals(self):
        # clear the canvas
        self.figure_1.clear()
        channel_1 = int(self.cb_plot_channel_1_process_signals.currentText())
        channel_2 = int(self.cb_plot_channel_2_process_signals.currentText())
        df = pd.read_csv("M_Class_Data/training_matrix_csv_m_class.csv")
        self.plot_values_of_channels_process_emg(df, channel_1, channel_2)
        # refresh canvas
        self.canvas_process_signals.draw()

    def plot_values_of_channels_process_emg(self, df, col1, col2):
        ax1 = self.figure_1.add_subplot(111)
        categories = df["Category"].unique()
        colors = ["red", "blue", "green", "yellow", "brown"]
        angles = ['180°', '90°', '65', '45', '30']
        for i, j in enumerate(categories):
            mask = df["Category"] == j
            df_x = df[mask]
            ax1.scatter(x=df_x[df.columns[col1 - 1]], y=df_x[df.columns[col2 - 1]], color=colors[i], label=angles[i])
            #pass

        #mask1 = df["Category"] == 0
        #mask2 = df["Category"] == 1
        #df_1 = df[mask1]
        #df_2 = df[mask2]
        #plt.scatter(x=df_1[df.columns[col1 - 1]], y=df_1[df.columns[col2 - 1]], color="red", label='180°')
        #plt.scatter(x=df_2[df.columns[col1 - 1]], y=df_2[df.columns[col2 - 1]], color="blue", label='90°')

        # Decorate
        ax1.set_title(f'Values in Channel_{col1} and Chanel_{col2} per category')
        ax1.set_xlabel(f'Chanel_{col1} - value')
        ax1.set_ylabel(f'Chane_{col2} - value')
        ax1.legend(loc='best')
        #plt.show()

    def plot_values_of_chanels_raw_emg(self, df, col1):
        ax2 = self.figure.add_subplot(111)
        x = np.arange(len(df))
        y = df[df.columns[col1 + 1]]
        ax2.plot(x, y)

        # Decorate
        ax2.set_title(f'Values in Channel_{col1} and the number of the sample')
        ax2.set_xlabel(f'Number of the sample - value')
        ax2.set_ylabel(f'Chanel_{col1} - value(mv)')


if __name__ == "__main__":



    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QWidget()
    ui = EmgApplication(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
