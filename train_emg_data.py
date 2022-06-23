# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'train_emg_data.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(640, 480)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtWidgets.QFrame(Form)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.home_btn = QtWidgets.QPushButton(self.frame)
        self.home_btn.setObjectName("home_btn")
        self.horizontalLayout.addWidget(self.home_btn)
        self.def_btn = QtWidgets.QPushButton(self.frame)
        self.def_btn.setObjectName("def_btn")
        self.horizontalLayout.addWidget(self.def_btn)
        self.results_btn = QtWidgets.QPushButton(self.frame)
        self.results_btn.setObjectName("results_btn")
        self.horizontalLayout.addWidget(self.results_btn)
        self.contact_btn = QtWidgets.QPushButton(self.frame)
        self.contact_btn.setObjectName("contact_btn")
        self.horizontalLayout.addWidget(self.contact_btn)
        self.verticalLayout.addWidget(self.frame)
        self.frame_2 = QtWidgets.QFrame(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pages = QtWidgets.QStackedWidget(self.frame_2)
        self.pages.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.pages.setObjectName("pages")
        self.home_pg = QtWidgets.QWidget()
        self.home_pg.setStyleSheet("background-color: rgb(64, 88, 191);")
        self.home_pg.setObjectName("home_pg")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.home_pg)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(self.home_pg)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.frame_3 = QtWidgets.QFrame(self.home_pg)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy)
        self.frame_3.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayout_15 = QtWidgets.QVBoxLayout(self.frame_3)
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.frame_14 = QtWidgets.QFrame(self.frame_3)
        self.frame_14.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_14.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_14.setObjectName("frame_14")
        self.horizontalLayout_22 = QtWidgets.QHBoxLayout(self.frame_14)
        self.horizontalLayout_22.setObjectName("horizontalLayout_22")
        self.treeWidget = QtWidgets.QTreeWidget(self.frame_14)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeWidget.sizePolicy().hasHeightForWidth())
        self.treeWidget.setSizePolicy(sizePolicy)
        self.treeWidget.setObjectName("treeWidget")
        item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)
        item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)
        item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)
        item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)
        item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)
        self.horizontalLayout_22.addWidget(self.treeWidget)
        self.frame_15 = QtWidgets.QFrame(self.frame_14)
        self.frame_15.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_15.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_15.setObjectName("frame_15")
        self.horizontalLayout_22.addWidget(self.frame_15)
        self.verticalLayout_15.addWidget(self.frame_14)
        self.train_all_signals = QtWidgets.QPushButton(self.frame_3)
        self.train_all_signals.setObjectName("train_all_signals")
        self.verticalLayout_15.addWidget(self.train_all_signals)
        self.verticalLayout_2.addWidget(self.frame_3)
        self.pages.addWidget(self.home_pg)
        self.def_pg = QtWidgets.QWidget()
        self.def_pg.setObjectName("def_pg")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.def_pg)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_2 = QtWidgets.QLabel(self.def_pg)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_4.addWidget(self.label_2)
        self.frame_4 = QtWidgets.QFrame(self.def_pg)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame_4)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.frame_5 = QtWidgets.QFrame(self.frame_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_5.sizePolicy().hasHeightForWidth())
        self.frame_5.setSizePolicy(sizePolicy)
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout(self.frame_5)
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.frame_6 = QtWidgets.QFrame(self.frame_5)
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.frame_6)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_17 = QtWidgets.QLabel(self.frame_6)
        self.label_17.setObjectName("label_17")
        self.verticalLayout_5.addWidget(self.label_17)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.frame_6)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.frequency = QtWidgets.QLineEdit(self.frame_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frequency.sizePolicy().hasHeightForWidth())
        self.frequency.setSizePolicy(sizePolicy)
        self.frequency.setObjectName("frequency")
        self.horizontalLayout_3.addWidget(self.frequency)
        self.verticalLayout_5.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_4 = QtWidgets.QLabel(self.frame_6)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        self.window_time = QtWidgets.QLineEdit(self.frame_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.window_time.sizePolicy().hasHeightForWidth())
        self.window_time.setSizePolicy(sizePolicy)
        self.window_time.setObjectName("window_time")
        self.horizontalLayout_4.addWidget(self.window_time)
        self.verticalLayout_5.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_5 = QtWidgets.QLabel(self.frame_6)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_5.addWidget(self.label_5)
        self.n_of_channels = QtWidgets.QComboBox(self.frame_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.n_of_channels.sizePolicy().hasHeightForWidth())
        self.n_of_channels.setSizePolicy(sizePolicy)
        self.n_of_channels.setObjectName("n_of_channels")
        self.n_of_channels.addItem("")
        self.n_of_channels.addItem("")
        self.n_of_channels.addItem("")
        self.n_of_channels.addItem("")
        self.n_of_channels.addItem("")
        self.n_of_channels.addItem("")
        self.n_of_channels.addItem("")
        self.n_of_channels.addItem("")
        self.horizontalLayout_5.addWidget(self.n_of_channels)
        self.verticalLayout_5.addLayout(self.horizontalLayout_5)
        self.verticalLayout_11.addWidget(self.frame_6)
        self.frame_7 = QtWidgets.QFrame(self.frame_5)
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.frame_7)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_16 = QtWidgets.QLabel(self.frame_7)
        self.label_16.setObjectName("label_16")
        self.verticalLayout_6.addWidget(self.label_16)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_6 = QtWidgets.QLabel(self.frame_7)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_6.addWidget(self.label_6)
        self.highpass_filter = QtWidgets.QLineEdit(self.frame_7)
        self.highpass_filter.setObjectName("highpass_filter")
        self.horizontalLayout_6.addWidget(self.highpass_filter)
        self.verticalLayout_6.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_7 = QtWidgets.QLabel(self.frame_7)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_7.addWidget(self.label_7)
        self.bandstop_filter = QtWidgets.QLineEdit(self.frame_7)
        self.bandstop_filter.setObjectName("bandstop_filter")
        self.horizontalLayout_7.addWidget(self.bandstop_filter)
        self.verticalLayout_6.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_8 = QtWidgets.QLabel(self.frame_7)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_8.addWidget(self.label_8)
        self.wav_filter = QtWidgets.QLineEdit(self.frame_7)
        self.wav_filter.setObjectName("wav_filter")
        self.horizontalLayout_8.addWidget(self.wav_filter)
        self.verticalLayout_6.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_10 = QtWidgets.QLabel(self.frame_7)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_9.addWidget(self.label_10)
        self.levels_of_wav_filter = QtWidgets.QComboBox(self.frame_7)
        self.levels_of_wav_filter.setObjectName("levels_of_wav_filter")
        self.levels_of_wav_filter.addItem("")
        self.levels_of_wav_filter.addItem("")
        self.levels_of_wav_filter.addItem("")
        self.levels_of_wav_filter.addItem("")
        self.horizontalLayout_9.addWidget(self.levels_of_wav_filter)
        self.verticalLayout_6.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_11 = QtWidgets.QLabel(self.frame_7)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_10.addWidget(self.label_11)
        self.layers_to_use = QtWidgets.QComboBox(self.frame_7)
        self.layers_to_use.setObjectName("layers_to_use")
        self.layers_to_use.addItem("")
        self.layers_to_use.addItem("")
        self.layers_to_use.addItem("")
        self.layers_to_use.addItem("")
        self.horizontalLayout_10.addWidget(self.layers_to_use)
        self.verticalLayout_6.addLayout(self.horizontalLayout_10)
        self.verticalLayout_11.addWidget(self.frame_7)
        self.horizontalLayout_17.addLayout(self.verticalLayout_11)
        self.verticalLayout_12 = QtWidgets.QVBoxLayout()
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.frame_8 = QtWidgets.QFrame(self.frame_5)
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.frame_8)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label_18 = QtWidgets.QLabel(self.frame_8)
        self.label_18.setObjectName("label_18")
        self.verticalLayout_7.addWidget(self.label_18)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.label_12 = QtWidgets.QLabel(self.frame_8)
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_11.addWidget(self.label_12)
        self.type_of_matrix = QtWidgets.QComboBox(self.frame_8)
        self.type_of_matrix.setObjectName("type_of_matrix")
        self.type_of_matrix.addItem("")
        self.type_of_matrix.addItem("")
        self.horizontalLayout_11.addWidget(self.type_of_matrix)
        self.verticalLayout_7.addLayout(self.horizontalLayout_11)
        self.verticalLayout_12.addWidget(self.frame_8)
        self.frame_9 = QtWidgets.QFrame(self.frame_5)
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.frame_9)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.label_19 = QtWidgets.QLabel(self.frame_9)
        self.label_19.setObjectName("label_19")
        self.verticalLayout_8.addWidget(self.label_19)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.label_9 = QtWidgets.QLabel(self.frame_9)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_12.addWidget(self.label_9)
        self.test_size = QtWidgets.QLineEdit(self.frame_9)
        self.test_size.setText("")
        self.test_size.setObjectName("test_size")
        self.horizontalLayout_12.addWidget(self.test_size)
        self.verticalLayout_8.addLayout(self.horizontalLayout_12)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.label_13 = QtWidgets.QLabel(self.frame_9)
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_13.addWidget(self.label_13)
        self.random_state = QtWidgets.QLineEdit(self.frame_9)
        self.random_state.setObjectName("random_state")
        self.horizontalLayout_13.addWidget(self.random_state)
        self.verticalLayout_8.addLayout(self.horizontalLayout_13)
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.label_14 = QtWidgets.QLabel(self.frame_9)
        self.label_14.setObjectName("label_14")
        self.horizontalLayout_14.addWidget(self.label_14)
        self.cv = QtWidgets.QLineEdit(self.frame_9)
        self.cv.setObjectName("cv")
        self.horizontalLayout_14.addWidget(self.cv)
        self.verticalLayout_8.addLayout(self.horizontalLayout_14)
        self.verticalLayout_12.addWidget(self.frame_9)
        self.frame_10 = QtWidgets.QFrame(self.frame_5)
        self.frame_10.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.frame_10)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.label_20 = QtWidgets.QLabel(self.frame_10)
        self.label_20.setObjectName("label_20")
        self.verticalLayout_9.addWidget(self.label_20)
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.label_15 = QtWidgets.QLabel(self.frame_10)
        self.label_15.setObjectName("label_15")
        self.horizontalLayout_16.addWidget(self.label_15)
        self.emg_file = QtWidgets.QLineEdit(self.frame_10)
        self.emg_file.setObjectName("emg_file")
        self.horizontalLayout_16.addWidget(self.emg_file)
        self.verticalLayout_9.addLayout(self.horizontalLayout_16)
        self.verticalLayout_12.addWidget(self.frame_10)
        self.horizontalLayout_17.addLayout(self.verticalLayout_12)
        self.verticalLayout_3.addWidget(self.frame_5)
        self.verticalLayout_4.addWidget(self.frame_4)
        self.pages.addWidget(self.def_pg)
        self.results_pg = QtWidgets.QWidget()
        self.results_pg.setObjectName("results_pg")
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout(self.results_pg)
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout()
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.plot_conf_matrix_btn = QtWidgets.QPushButton(self.results_pg)
        self.plot_conf_matrix_btn.setObjectName("plot_conf_matrix_btn")
        self.verticalLayout_13.addWidget(self.plot_conf_matrix_btn)
        self.plot_roc_curve_btn = QtWidgets.QPushButton(self.results_pg)
        self.plot_roc_curve_btn.setObjectName("plot_roc_curve_btn")
        self.verticalLayout_13.addWidget(self.plot_roc_curve_btn)
        self.plot_class_scores_btn = QtWidgets.QPushButton(self.results_pg)
        self.plot_class_scores_btn.setObjectName("plot_class_scores_btn")
        self.verticalLayout_13.addWidget(self.plot_class_scores_btn)
        self.plot_raw_signals_btn = QtWidgets.QPushButton(self.results_pg)
        self.plot_raw_signals_btn.setObjectName("plot_raw_signals_btn")
        self.verticalLayout_13.addWidget(self.plot_raw_signals_btn)
        self.plot_process_signals_btn = QtWidgets.QPushButton(self.results_pg)
        self.plot_process_signals_btn.setObjectName("plot_process_signals_btn")
        self.verticalLayout_13.addWidget(self.plot_process_signals_btn)
        self.horizontalLayout_15.addLayout(self.verticalLayout_13)
        self.results_pgs = QtWidgets.QStackedWidget(self.results_pg)
        self.results_pgs.setStyleSheet("background-color: rgb(138, 226, 52);")
        self.results_pgs.setObjectName("results_pgs")
        self.confusion_pg = QtWidgets.QWidget()
        self.confusion_pg.setObjectName("confusion_pg")
        self.label_21 = QtWidgets.QLabel(self.confusion_pg)
        self.label_21.setGeometry(QtCore.QRect(190, 20, 121, 17))
        self.label_21.setObjectName("label_21")
        self.results_pgs.addWidget(self.confusion_pg)
        self.roc_pg = QtWidgets.QWidget()
        self.roc_pg.setObjectName("roc_pg")
        self.label_22 = QtWidgets.QLabel(self.roc_pg)
        self.label_22.setGeometry(QtCore.QRect(200, 20, 91, 17))
        self.label_22.setObjectName("label_22")
        self.results_pgs.addWidget(self.roc_pg)
        self.classification_scores_pg = QtWidgets.QWidget()
        self.classification_scores_pg.setObjectName("classification_scores_pg")
        self.label_23 = QtWidgets.QLabel(self.classification_scores_pg)
        self.label_23.setGeometry(QtCore.QRect(170, 20, 141, 16))
        self.label_23.setObjectName("label_23")
        self.frame_11 = QtWidgets.QFrame(self.classification_scores_pg)
        self.frame_11.setGeometry(QtCore.QRect(10, 160, 431, 251))
        self.frame_11.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.frame_11.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_11.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_11.setObjectName("frame_11")
        self.results_pgs.addWidget(self.classification_scores_pg)
        self.plot_raw_signals_pg = QtWidgets.QWidget()
        self.plot_raw_signals_pg.setObjectName("plot_raw_signals_pg")
        self.verticalLayout_14 = QtWidgets.QVBoxLayout(self.plot_raw_signals_pg)
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.label_24 = QtWidgets.QLabel(self.plot_raw_signals_pg)
        self.label_24.setObjectName("label_24")
        self.verticalLayout_14.addWidget(self.label_24)
        self.horizontalLayout_23 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_23.setObjectName("horizontalLayout_23")
        self.horizontalLayout_21 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_21.setObjectName("horizontalLayout_21")
        self.label_29 = QtWidgets.QLabel(self.plot_raw_signals_pg)
        self.label_29.setObjectName("label_29")
        self.horizontalLayout_21.addWidget(self.label_29)
        self.cb_raw_signal_1 = QtWidgets.QComboBox(self.plot_raw_signals_pg)
        self.cb_raw_signal_1.setObjectName("cb_raw_signal_1")
        self.cb_raw_signal_1.addItem("")
        self.cb_raw_signal_1.addItem("")
        self.cb_raw_signal_1.addItem("")
        self.cb_raw_signal_1.addItem("")
        self.cb_raw_signal_1.addItem("")
        self.cb_raw_signal_1.addItem("")
        self.cb_raw_signal_1.addItem("")
        self.cb_raw_signal_1.addItem("")
        self.horizontalLayout_21.addWidget(self.cb_raw_signal_1)
        self.horizontalLayout_23.addLayout(self.horizontalLayout_21)
        self.verticalLayout_14.addLayout(self.horizontalLayout_23)
        self.frame_12 = QtWidgets.QFrame(self.plot_raw_signals_pg)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_12.sizePolicy().hasHeightForWidth())
        self.frame_12.setSizePolicy(sizePolicy)
        self.frame_12.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.frame_12.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_12.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_12.setObjectName("frame_12")
        self.verticalLayout_14.addWidget(self.frame_12)
        self.plot_raw_signals_btn_2 = QtWidgets.QPushButton(self.plot_raw_signals_pg)
        self.plot_raw_signals_btn_2.setObjectName("plot_raw_signals_btn_2")
        self.verticalLayout_14.addWidget(self.plot_raw_signals_btn_2)
        self.results_pgs.addWidget(self.plot_raw_signals_pg)
        self.plot_processed_signals_pg = QtWidgets.QWidget()
        self.plot_processed_signals_pg.setObjectName("plot_processed_signals_pg")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.plot_processed_signals_pg)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.label_25 = QtWidgets.QLabel(self.plot_processed_signals_pg)
        self.label_25.setObjectName("label_25")
        self.verticalLayout_10.addWidget(self.label_25)
        self.horizontalLayout_20 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_20.setObjectName("horizontalLayout_20")
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.label_27 = QtWidgets.QLabel(self.plot_processed_signals_pg)
        self.label_27.setObjectName("label_27")
        self.horizontalLayout_18.addWidget(self.label_27)
        self.cb_plot_channel_1_process_signals = QtWidgets.QComboBox(self.plot_processed_signals_pg)
        self.cb_plot_channel_1_process_signals.setObjectName("cb_plot_channel_1_process_signals")
        self.cb_plot_channel_1_process_signals.addItem("")
        self.cb_plot_channel_1_process_signals.addItem("")
        self.cb_plot_channel_1_process_signals.addItem("")
        self.cb_plot_channel_1_process_signals.addItem("")
        self.cb_plot_channel_1_process_signals.addItem("")
        self.cb_plot_channel_1_process_signals.addItem("")
        self.cb_plot_channel_1_process_signals.addItem("")
        self.cb_plot_channel_1_process_signals.addItem("")
        self.horizontalLayout_18.addWidget(self.cb_plot_channel_1_process_signals)
        self.horizontalLayout_20.addLayout(self.horizontalLayout_18)
        self.horizontalLayout_19 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_19.setObjectName("horizontalLayout_19")
        self.label_28 = QtWidgets.QLabel(self.plot_processed_signals_pg)
        self.label_28.setObjectName("label_28")
        self.horizontalLayout_19.addWidget(self.label_28)
        self.cb_plot_channel_2_process_signals = QtWidgets.QComboBox(self.plot_processed_signals_pg)
        self.cb_plot_channel_2_process_signals.setObjectName("cb_plot_channel_2_process_signals")
        self.cb_plot_channel_2_process_signals.addItem("")
        self.cb_plot_channel_2_process_signals.addItem("")
        self.cb_plot_channel_2_process_signals.addItem("")
        self.cb_plot_channel_2_process_signals.addItem("")
        self.cb_plot_channel_2_process_signals.addItem("")
        self.cb_plot_channel_2_process_signals.addItem("")
        self.cb_plot_channel_2_process_signals.addItem("")
        self.cb_plot_channel_2_process_signals.addItem("")
        self.horizontalLayout_19.addWidget(self.cb_plot_channel_2_process_signals)
        self.horizontalLayout_20.addLayout(self.horizontalLayout_19)
        self.verticalLayout_10.addLayout(self.horizontalLayout_20)
        self.frame_13 = QtWidgets.QFrame(self.plot_processed_signals_pg)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_13.sizePolicy().hasHeightForWidth())
        self.frame_13.setSizePolicy(sizePolicy)
        self.frame_13.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.frame_13.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_13.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_13.setObjectName("frame_13")
        self.verticalLayout_10.addWidget(self.frame_13)
        self.btn_plot_processed_signals = QtWidgets.QPushButton(self.plot_processed_signals_pg)
        self.btn_plot_processed_signals.setObjectName("btn_plot_processed_signals")
        self.verticalLayout_10.addWidget(self.btn_plot_processed_signals)
        self.results_pgs.addWidget(self.plot_processed_signals_pg)
        self.horizontalLayout_15.addWidget(self.results_pgs)
        self.pages.addWidget(self.results_pg)
        self.contact_pg = QtWidgets.QWidget()
        self.contact_pg.setObjectName("contact_pg")
        self.label_26 = QtWidgets.QLabel(self.contact_pg)
        self.label_26.setGeometry(QtCore.QRect(300, 20, 111, 17))
        self.label_26.setObjectName("label_26")
        self.pages.addWidget(self.contact_pg)
        self.horizontalLayout_2.addWidget(self.pages)
        self.verticalLayout.addWidget(self.frame_2)

        self.retranslateUi(Form)
        self.pages.setCurrentIndex(0)
        self.results_pgs.setCurrentIndex(4)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.home_btn.setText(_translate("Form", "Home"))
        self.def_btn.setText(_translate("Form", "Definitions"))
        self.results_btn.setText(_translate("Form", "Results"))
        self.contact_btn.setText(_translate("Form", "Contact"))
        self.label.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-size:20pt; color:#00ff00;\">Training EMG Signals</span></p></body></html>"))
        self.treeWidget.headerItem().setText(0, _translate("Form", "Classifier"))
        self.treeWidget.headerItem().setText(1, _translate("Form", "Score"))
        __sortingEnabled = self.treeWidget.isSortingEnabled()
        self.treeWidget.setSortingEnabled(False)
        self.treeWidget.topLevelItem(0).setText(0, _translate("Form", "LDA"))
        self.treeWidget.topLevelItem(0).setText(1, _translate("Form", "Not trained"))
        self.treeWidget.topLevelItem(1).setText(0, _translate("Form", "GNB"))
        self.treeWidget.topLevelItem(1).setText(1, _translate("Form", "Not trained"))
        self.treeWidget.topLevelItem(2).setText(0, _translate("Form", "Lin_SVM"))
        self.treeWidget.topLevelItem(2).setText(1, _translate("Form", "Not trained"))
        self.treeWidget.topLevelItem(3).setText(0, _translate("Form", "Knn"))
        self.treeWidget.topLevelItem(3).setText(1, _translate("Form", "Not trained"))
        self.treeWidget.topLevelItem(4).setText(0, _translate("Form", "Tree"))
        self.treeWidget.topLevelItem(4).setText(1, _translate("Form", "Not trained"))
        self.treeWidget.setSortingEnabled(__sortingEnabled)
        self.train_all_signals.setText(_translate("Form", "Train the signals"))
        self.label_2.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-size:18pt; font-weight:600;\">General Definitions</span></p></body></html>"))
        self.label_17.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Basic Definitions</span></p></body></html>"))
        self.label_3.setText(_translate("Form", "Frequency"))
        self.frequency.setPlaceholderText(_translate("Form", "200"))
        self.label_4.setText(_translate("Form", "Window Time"))
        self.window_time.setPlaceholderText(_translate("Form", "250"))
        self.label_5.setText(_translate("Form", "N° of Channels"))
        self.n_of_channels.setItemText(0, _translate("Form", "8"))
        self.n_of_channels.setItemText(1, _translate("Form", "1"))
        self.n_of_channels.setItemText(2, _translate("Form", "2"))
        self.n_of_channels.setItemText(3, _translate("Form", "3"))
        self.n_of_channels.setItemText(4, _translate("Form", "4"))
        self.n_of_channels.setItemText(5, _translate("Form", "5"))
        self.n_of_channels.setItemText(6, _translate("Form", "6"))
        self.n_of_channels.setItemText(7, _translate("Form", "7"))
        self.label_16.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Filtering Definitions</span></p></body></html>"))
        self.label_6.setText(_translate("Form", "Highpass filter"))
        self.label_7.setText(_translate("Form", "bandstop filter"))
        self.label_8.setText(_translate("Form", "wav_filter"))
        self.wav_filter.setPlaceholderText(_translate("Form", "db7"))
        self.label_10.setText(_translate("Form", "levels of wav filter"))
        self.levels_of_wav_filter.setItemText(0, _translate("Form", "2"))
        self.levels_of_wav_filter.setItemText(1, _translate("Form", "1"))
        self.levels_of_wav_filter.setItemText(2, _translate("Form", "3"))
        self.levels_of_wav_filter.setItemText(3, _translate("Form", "4"))
        self.label_11.setText(_translate("Form", "layers to use"))
        self.layers_to_use.setItemText(0, _translate("Form", "4"))
        self.layers_to_use.setItemText(1, _translate("Form", "1"))
        self.layers_to_use.setItemText(2, _translate("Form", "2"))
        self.layers_to_use.setItemText(3, _translate("Form", "3"))
        self.label_18.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Matrix Definition</span></p></body></html>"))
        self.label_12.setText(_translate("Form", "type of matrix"))
        self.type_of_matrix.setItemText(0, _translate("Form", "rms"))
        self.type_of_matrix.setItemText(1, _translate("Form", "mav"))
        self.label_19.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Training Definitions</span></p></body></html>"))
        self.label_9.setText(_translate("Form", "test size"))
        self.test_size.setPlaceholderText(_translate("Form", "0.3"))
        self.label_13.setText(_translate("Form", "Random State"))
        self.random_state.setPlaceholderText(_translate("Form", "42"))
        self.label_14.setText(_translate("Form", "cv"))
        self.cv.setPlaceholderText(_translate("Form", "5"))
        self.label_20.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Raw Data EMG</span></p></body></html>"))
        self.label_15.setText(_translate("Form", "EMG file"))
        self.plot_conf_matrix_btn.setText(_translate("Form", "Confusion Matrix"))
        self.plot_roc_curve_btn.setText(_translate("Form", "ROC Curve"))
        self.plot_class_scores_btn.setText(_translate("Form", "Classification Scores"))
        self.plot_raw_signals_btn.setText(_translate("Form", "Plot Raw Signals"))
        self.plot_process_signals_btn.setText(_translate("Form", "Plot Processed Signals"))
        self.label_21.setText(_translate("Form", "Confusion Matrix"))
        self.label_22.setText(_translate("Form", "ROC Curve"))
        self.label_23.setText(_translate("Form", "Classification Scores"))
        self.label_24.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Plot Raw Signals</span></p></body></html>"))
        self.label_29.setText(_translate("Form", "Channel_1"))
        self.cb_raw_signal_1.setItemText(0, _translate("Form", "1"))
        self.cb_raw_signal_1.setItemText(1, _translate("Form", "2"))
        self.cb_raw_signal_1.setItemText(2, _translate("Form", "3"))
        self.cb_raw_signal_1.setItemText(3, _translate("Form", "4"))
        self.cb_raw_signal_1.setItemText(4, _translate("Form", "5"))
        self.cb_raw_signal_1.setItemText(5, _translate("Form", "6"))
        self.cb_raw_signal_1.setItemText(6, _translate("Form", "7"))
        self.cb_raw_signal_1.setItemText(7, _translate("Form", "8"))
        self.plot_raw_signals_btn_2.setText(_translate("Form", "Plot Raw Signals"))
        self.label_25.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Plot Processed Signals</span></p></body></html>"))
        self.label_27.setText(_translate("Form", "Channel_1"))
        self.cb_plot_channel_1_process_signals.setItemText(0, _translate("Form", "1"))
        self.cb_plot_channel_1_process_signals.setItemText(1, _translate("Form", "2"))
        self.cb_plot_channel_1_process_signals.setItemText(2, _translate("Form", "3"))
        self.cb_plot_channel_1_process_signals.setItemText(3, _translate("Form", "4"))
        self.cb_plot_channel_1_process_signals.setItemText(4, _translate("Form", "5"))
        self.cb_plot_channel_1_process_signals.setItemText(5, _translate("Form", "6"))
        self.cb_plot_channel_1_process_signals.setItemText(6, _translate("Form", "7"))
        self.cb_plot_channel_1_process_signals.setItemText(7, _translate("Form", "8"))
        self.label_28.setText(_translate("Form", "Channel_2"))
        self.cb_plot_channel_2_process_signals.setItemText(0, _translate("Form", "1"))
        self.cb_plot_channel_2_process_signals.setItemText(1, _translate("Form", "2"))
        self.cb_plot_channel_2_process_signals.setItemText(2, _translate("Form", "3"))
        self.cb_plot_channel_2_process_signals.setItemText(3, _translate("Form", "4"))
        self.cb_plot_channel_2_process_signals.setItemText(4, _translate("Form", "5"))
        self.cb_plot_channel_2_process_signals.setItemText(5, _translate("Form", "6"))
        self.cb_plot_channel_2_process_signals.setItemText(6, _translate("Form", "7"))
        self.cb_plot_channel_2_process_signals.setItemText(7, _translate("Form", "8"))
        self.btn_plot_processed_signals.setText(_translate("Form", "Plot Signals"))
        self.label_26.setText(_translate("Form", "Contact Page"))
