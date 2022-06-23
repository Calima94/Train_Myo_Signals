from train_emg_data import Ui_Form
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
import sys
from matplotlib.figure import Figure
from numpy import random
import pandas as pd
import numpy as np
import train_signals_emg


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=4, height=5, dpi=200):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(100)
        super(MplCanvas, self).__init__(fig)


class EmgApplication(Ui_Form):
    def __init__(self, definitions):
        super(Ui_Form, self).__init__()
        self.setupUi(definitions)
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

    def train_myo(self):
        train_signals_emg.train_signals_emg()


    def set_of_canvas(self):
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
