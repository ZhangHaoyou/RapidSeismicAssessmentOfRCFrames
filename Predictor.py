import sys
import sip
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_frame import plot_frame
from matplotlib import rcParams, ticker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QComboBox, QWidget, QLabel, QLineEdit, QPushButton, QGridLayout, QGroupBox, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import pyqtSlot, Qt, QSize
from PyQt5 import QtGui


config = {
    "font.family":'Arial',
    "font.size": 15,
    # "mathtext.fontset":'stix',
}
rcParams.update(config)

data   = pd.read_excel("./Data/DamageState9900_detailed.xls", sheet_name="Sheet1")
X_list = ["Story num", "num_short_span", "length_short_span", "num_long_span", "length_long_span", "story_height", "PGA(g)", "PGV",
    "Sa(0.2s)(g)", "Sa(0.4s)(g)", "Sa(0.6s)(g)", "Sa(0.8s)(g)", "Sa(1.0s)(g)", 
    "Sa(1.2s)(g)",  "Sa(1.4s)(g)", "Sa(1.6s)(g)", "Sa(1.8s)(g)", "Sa(2.0s)(g)", 
    "EPA(g)", "HI", 
    "Intensity", "year type"]
X = data.loc[:, X_list].to_numpy()
y = data["DS"].to_numpy().astype('int64')
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=2956)
X_train, X_val, y_train, y_val           = train_test_split(X_train_val, y_train_val, test_size=0.1/(1-0.1), random_state=2956)

Sscaler = StandardScaler()
X_train = Sscaler.fit_transform(X_train)
Mscaler = MinMaxScaler()
X_train = Mscaler.fit_transform(X_train)

gm    = np.loadtxt("./Data/GroundMotion.txt", delimiter="\t")
model = pickle.load(open('./Data/model_QBC400_detailed.pkl','rb'))

dpi   = 170
font_size_label = 8
font_size_tick  = 8

def year_clf(year):
    year = int(year)
    if year<=1978:
        return 0
    elif year<=1989:
        return 0
    elif year<=2001:
        return 1
    else:
        return 2

def normalize(original_data, Sscaler=Sscaler, Mscaler=Mscaler):
    normalized_standard = Sscaler.transform(original_data)
    normalized_min_max  = Mscaler.transform(normalized_standard)
    return normalized_min_max

def toDS(DS):
    DS = int(DS)
    if DS == 0:
        return "None"
    elif DS == 1:
        return "Slight"
    elif DS == 2:
        return "Moderate"
    elif DS == 3:
        return "Extensive"
    elif DS == 4:
        return "Complete"
    else:
        return "Out of range [0, 4]!!!"

def plot_event(gm=gm, figsize=(3.48,1), path="./gm.jpg", font_size_label=font_size_label, labelsize=font_size_tick):
    x = gm[:,0]
    y = gm[:,1]
    y_max = np.argmax(y)
    y_min = np.argmin(y)
    max_val = y[y_max]
    min_val = y[y_min]
    if max_val>abs(min_val):
        annotate_y = max_val
        annotate_x = x[y_max]
    else:
        annotate_y = min_val
        annotate_x = x[y_min]

    plt.figure(figsize=figsize)
    plt.plot(x, y, "-b")
    plt.tick_params(labelsize=labelsize)
    plt.xlabel("Time (s)", fontsize=font_size_label)
    plt.ylabel("Acceleration (gal)", fontsize=font_size_label)
    plt.annotate("PGA = {}".format(annotate_y), xy=(annotate_x, annotate_y),
        xytext=(annotate_x+2, annotate_y-100), fontsize=8)
    plt.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close("all")

def plot_prob(probas=np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]), figsize=(3.0,1.5), path="./probability.jpg",
    font_size_label=font_size_label, labelsize=font_size_tick, color="blue"):
    plt.figure(figsize=figsize)
    plt.barh(list(range(0,5)), list(probas.ravel()), color=color)
    plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1,decimals=0))
    plt.yticks(list(range(0,5)), ["None", "Slight", "Moderate", "Extensive", "Complete"])
    plt.xlim([0,1])
    plt.tick_params(labelsize=labelsize)
    plt.ylabel("Damage state", fontsize=font_size_label)
    plt.xlabel("Probability", fontsize=font_size_label)
    for x, y in zip(list(range(0,5)), list(probas.ravel())):
        plt.annotate(format(y, ".0%"), xy=(y,x), xytext=(y,x-0.15), fontsize=8)
    plt.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close("all")

class predictor(QWidget):

    def __init__(self):
        super(predictor, self).__init__()
        self.initUI()
        self.setWindowTitle('Rapid Seismic Damage State Assessment of RC Frames ')
        
        self.setWindowIcon(QtGui.QIcon(sys.path[0]+'/logo.jpg'))

    def initUI(self):
        # subtitle
        self.intro_label = QLabel('Rapid Seismic Damage State Assessment of RC Frames ')
        self.intro_label.setAlignment(Qt.AlignCenter)
        self.intro_label.setStyleSheet('color:rgb(0,150,195); font-weight:bold; background-color:orange; border-radius:10px; border:2px groove gray; border-style:outset')
        self.intro_label.setFixedHeight(50)

        event_name_list = ['2016-02-06-Meinong-Taiwan-China', '2022-02-06-Meinong-Taiwan-China']
        # parameters
        self.ns_label  = QLabel('Number of stories', self)
        self.hs_label  = QLabel('Story height', self)
        self.nss_label = QLabel('Number of bays on the shorter side', self)
        self.lss_label = QLabel('Length of the bay on the shorter side', self)
        self.nls_label = QLabel('Number of bays on the longer side', self)
        self.lls_label = QLabel('Length of the bay on the longer side', self)
        self.cp_label  = QLabel('Constructed period', self)
        self.sdi_label = QLabel('Seismic design intensity', self)
        self.lat_label = QLabel('Latitude', self)
        self.lon_label = QLabel('Longitude', self)
        self.en_label  = QLabel('Event name', self)
        
        self.ds_label   = QLabel('Damage State', self)
        self.prob_label       = QLabel('Probability', self)

        # input box
        self.ns_line = QLineEdit('3', self)
        self.ns_line.setStyleSheet('background-color:white; border-radius:10px')
        # self.ns_line.setFixedWidth(323)
        self.ns_line.setFixedWidth(210)
        # self.ns_line.editingFinished.connect(self.on_frame_change)
        self.hs_line     = QLineEdit('3.33', self)
        self.hs_line.setStyleSheet('background-color:white; border-radius:10px')
        self.hs_line.editingFinished.connect(self.on_frame_change)
        self.nss_line    = QLineEdit('2', self)
        self.nss_line.setStyleSheet('background-color:white; border-radius:10px')
        self.nss_line.editingFinished.connect(self.on_frame_change)
        self.lss_line    = QLineEdit('5.4', self)
        self.lss_line.setStyleSheet('background-color:white; border-radius:10px')
        self.lss_line.editingFinished.connect(self.on_frame_change)
        self.nls_line     = QLineEdit('4', self)
        self.nls_line.setStyleSheet('background-color:white; border-radius:10px')
        self.nls_line.editingFinished.connect(self.on_frame_change)
        self.lls_line = QLineEdit('4.1', self)
        self.lls_line.setStyleSheet('background-color:white; border-radius:10px')
        self.lls_line.editingFinished.connect(self.on_frame_change)
        self.cp_line  = QLineEdit('1990', self)
        self.cp_line.setStyleSheet('background-color:white; border-radius:10px')
        self.sdi_line  = QLineEdit('8', self)
        self.sdi_line.setStyleSheet('background-color:white; border-radius:10px')
        self.lat_line          = QLineEdit('23.11980556', self)
        self.lat_line.setStyleSheet('background-color:white; border-radius:10px')
        self.lon_line          = QLineEdit('120.4676944', self)
        self.lon_line.setStyleSheet('background-color:white; border-radius:10px')
        self.en_combox = QComboBox(self)
        self.en_combox.setStyleSheet('background-color:white; border-radius:10px')
        self.en_combox.addItems(event_name_list)
        self.en_combox.currentTextChanged.connect(self.on_event_change)
        self.ds_line   = QLineEdit(self)
        self.ds_line.setStyleSheet('background-color:white; border-radius:10px')
        self.prob_line       = QLineEdit(self)
        self.prob_line.setStyleSheet('background-color:white; border-radius:10px')

        # button
        self.pred_button = QPushButton('Predict', self)
        self.pred_button.clicked.connect(self.on_pred_button_click)
        self.pred_button.setStyleSheet('color:red; background-color:rgb(0,150,195); border-radius:10px; border:2px groove gray; border-style:outset;')

        # plot
        # plot_frame()
        self.frame_fig_label= QLabel(self)
        self.frame_fig = QtGui.QPixmap()
        self.frame_fig.load(sys.path[0]+'\\frame_croped.jpg')
        self.frame_fig_label.setPixmap(self.frame_fig)
        self.frame_fig_label.setAlignment(Qt.AlignCenter)
        # self.frame_fig_label.setScaledContents(True)
        plot_event()
        self.event_fig_label= QLabel(self)
        self.event_fig = QtGui.QPixmap()
        self.event_fig.load(sys.path[0]+'\\gm.jpg')
        self.event_fig_label.setPixmap(self.event_fig)
        self.event_fig_label.setAlignment(Qt.AlignCenter)
        # self.event_fig_label.setScaledContents(True)
        plot_prob()
        self.prob_fig_label= QLabel(self)
        self.prob_fig = QtGui.QPixmap()
        self.prob_fig.load(sys.path[0]+'\\probability.jpg')
        self.prob_fig_label.setPixmap(self.prob_fig)
        self.prob_fig_label.setAlignment(Qt.AlignCenter)
        # self.prob_fig_label.setScaledContents(True)

        # layout
        self.grid_layout_intro  = QGridLayout()
        self.grid_layout_build  = QGridLayout()
        self.grid_layout_event = QGridLayout()
        self.grid_layout_output   = QGridLayout()
        self.grid_layout_pred   = QGridLayout()
        self.grid_layout_frame   = QGridLayout()
        self.grid_layout_gm_fig   = QGridLayout()
        self.grid_layout_prob   = QGridLayout()
        
        spacing = 23
        self.grid_layout_intro.addWidget(self.intro_label, 0, 0, 1, 3)
        self.row_start_build = 2
        self.grid_layout_build.addWidget(self.ns_label, self.row_start_build+0, 0, 1, 1)
        self.grid_layout_build.addWidget(self.ns_line, self.row_start_build+0, 1, 1, 1)
        self.grid_layout_build.addWidget(self.hs_label, self.row_start_build+1, 0, 1, 1)
        self.grid_layout_build.addWidget(self.hs_line, self.row_start_build+1, 1, 1, 1)
        self.grid_layout_build.addWidget(self.nss_label, self.row_start_build+2, 0, 1, 1)
        self.grid_layout_build.addWidget(self.nss_line, self.row_start_build+2, 1, 1, 1)
        self.grid_layout_build.addWidget(self.lss_label, self.row_start_build+3, 0, 1, 1)
        self.grid_layout_build.addWidget(self.lss_line, self.row_start_build+3, 1, 1, 1)
        self.grid_layout_build.addWidget(self.nls_label, self.row_start_build+4, 0, 1, 1)
        self.grid_layout_build.addWidget(self.nls_line, self.row_start_build+4, 1, 1, 1)
        self.grid_layout_build.addWidget(self.lls_label, self.row_start_build+5, 0, 1, 1)
        self.grid_layout_build.addWidget(self.lls_line, self.row_start_build+5, 1, 1, 1)
        self.grid_layout_build.addWidget(self.cp_label, self.row_start_build+6, 0, 1, 1)
        self.grid_layout_build.addWidget(self.cp_line, self.row_start_build+6, 1, 1, 1)
        self.grid_layout_build.addWidget(self.sdi_label, self.row_start_build+7, 0, 1, 1)
        self.grid_layout_build.addWidget(self.sdi_line, self.row_start_build+7, 1, 1, 1)
        self.grid_layout_build.addWidget(self.lat_label, self.row_start_build+8, 0, 1, 1)
        self.grid_layout_build.addWidget(self.lat_line, self.row_start_build+8, 1, 1, 1)
        self.grid_layout_build.addWidget(self.lon_label, self.row_start_build+9, 0, 1, 1)
        self.grid_layout_build.addWidget(self.lon_line, self.row_start_build+9, 1, 1, 1)
        self.grid_layout_build.setSpacing(spacing)
        self.row_start_event = 12
        self.grid_layout_event.addWidget(self.en_label, self.row_start_event+0, 0, 1, 1)
        self.grid_layout_event.addWidget(self.en_combox, self.row_start_event+0, 1, 1, 1)
        self.grid_layout_event.setSpacing(spacing)
        self.row_start_output = 14
        self.grid_layout_output.addWidget(self.ds_label, self.row_start_output+0, 0, 1, 1)
        self.grid_layout_output.addWidget(self.ds_line, self.row_start_output+0, 1, 1, 1)
        self.grid_layout_output.addWidget(self.prob_label, self.row_start_output+1, 0, 1, 1)
        self.grid_layout_output.addWidget(self.prob_line, self.row_start_output+1, 1, 1, 1)
        self.grid_layout_output.setSpacing(spacing)
        self.row_start_pred = 16
        self.grid_layout_pred.addWidget(self.pred_button, self.row_start_pred+0, 0, 1, 2)
        self.row_start_frame = self.row_start_build
        self.grid_layout_frame.addWidget(self.frame_fig_label, self.row_start_frame+0, 1, 8, 2)
        self.row_start_gm_fig = self.row_start_build
        self.grid_layout_gm_fig.addWidget(self.event_fig_label, self.row_start_event+8, 1, 2, 2)
        self.row_start_prob = self.row_start_build
        self.grid_layout_prob.addWidget(self.prob_fig_label, self.row_start_prob+10, 1, 6, 2)

        self.groupbox_build  = QGroupBox('Building information', self)
        self.groupbox_build.setStyleSheet('QGroupBox:title {color: rgb(0,150,195);}')
        self.groupbox_build.setLayout(self.grid_layout_build)
        self.groupbox_event  = QGroupBox('Event information', self)
        self.groupbox_event.setStyleSheet('QGroupBox:title {color: rgb(0,150,195);}')
        self.groupbox_event.setLayout(self.grid_layout_event)
        self.groupbox_output = QGroupBox('Output', self)
        self.groupbox_output.setStyleSheet('QGroupBox:title {color: rgb(0,150,195);}')
        self.groupbox_output.setLayout(self.grid_layout_output)

        self.vbox_layout_var = QVBoxLayout()
        self.vbox_layout_plot = QVBoxLayout()
        self.hbox_layout_var_plot = QHBoxLayout()
        self.vbox_layout_all = QVBoxLayout()
        self.vbox_layout_var.addWidget(self.groupbox_build)
        self.vbox_layout_var.addWidget(self.groupbox_event)
        self.vbox_layout_var.addWidget(self.groupbox_output)
        self.vbox_layout_var.addLayout(self.grid_layout_pred)
        self.vbox_layout_plot.addLayout(self.grid_layout_frame)
        self.vbox_layout_plot.addLayout(self.grid_layout_gm_fig)
        self.vbox_layout_plot.addLayout(self.grid_layout_prob)
        self.hbox_layout_var_plot.addLayout(self.vbox_layout_var)
        self.hbox_layout_var_plot.addLayout(self.vbox_layout_plot)
        self.vbox_layout_all.addLayout(self.grid_layout_intro)
        self.vbox_layout_all.addLayout(self.hbox_layout_var_plot)

        self.setLayout(self.vbox_layout_all)
    
    @pyqtSlot()
    def on_frame_change(self):
        self.ns_val  = int(self.ns_line.text())
        self.hs_val  = float(self.hs_line.text())
        self.nss_val = int(self.nss_line.text())
        self.lss_val = float(self.lss_line.text())
        self.nls_val = int(self.nls_line.text())
        self.lls_val = float(self.lls_line.text())
        self.cp_val  = int(self.cp_line.text())
        self.sdi_val = int(self.sdi_line.text())
        self.lat_val = float(self.lat_line.text())
        self.lon_val = float(self.lon_line.text())

        plot_frame(ns=self.ns_val, hs=self.hs_val, nss=self.nss_val, lss=self.lss_val, nls=self.nls_val, lls=self.lls_val)
        self.frame_fig.load(sys.path[0]+'\\frame_croped.jpg')
        self.frame_fig_label.setPixmap(self.frame_fig)
        self.frame_fig_label.setAlignment(Qt.AlignCenter)
        # self.frame_fig_label.setScaledContents(True)
    
    @pyqtSlot()
    def on_event_change(self):
        self.ns_val  = int(self.ns_line.text())
        self.hs_val  = float(self.hs_line.text())
        self.nss_val = int(self.nss_line.text())
        self.lss_val = float(self.lss_line.text())
        self.nls_val = int(self.nls_line.text())
        self.lls_val = float(self.lls_line.text())
        self.cp_val  = int(self.cp_line.text())
        self.sdi_val = int(self.sdi_line.text())
        self.lat_val = float(self.lat_line.text())
        self.lon_val = float(self.lon_line.text())

        plot_event(gm=gm)
        self.event_fig.load(sys.path[0]+'\\gm.jpg')
        self.event_fig_label.setPixmap(self.event_fig)
        self.event_fig_label.setAlignment(Qt.AlignCenter)
        # self.event_fig_label.setScaledContents(True)

    @pyqtSlot()
    def on_pred_button_click(self):
        self.ns_val  = int(self.ns_line.text())
        self.hs_val  = float(self.hs_line.text())
        self.nss_val = int(self.nss_line.text())
        self.lss_val = float(self.lss_line.text())
        self.nls_val = int(self.nls_line.text())
        self.lls_val = float(self.lls_line.text())
        self.cp_val  = int(self.cp_line.text())
        self.sdi_val = int(self.sdi_line.text())
        self.lat_val = float(self.lat_line.text())
        self.lon_val = float(self.lon_line.text())

        # predict damage state
        year_type = year_clf(self.cp_val)
        features   = [np.array([self.ns_val, self.nss_val, self.lss_val, self.nls_val, self.lls_val, self.hs_val,
            0.453611224,20.1833775,1.168311973,0.405153339,0.556825148,0.480836675,0.33552279,0.243752801,0.166780751,0.140246614,0.112163787,0.080923306,0.136050715,111.396219,
                self.sdi_val, year_type])]
        features_norm = normalize(features)
        pred          = model.predict(features_norm)
        probas        = model.predict_proba(features_norm)
        proba0        = format(probas[0,0], ".2%")
        proba1        = format(probas[0,1], ".2%")
        proba2        = format(probas[0,2], ".2%")
        proba3        = format(probas[0,3], ".2%")
        proba4        = format(probas[0,4], ".2%")
        pred_proba    = format(probas.max(), ".2%")
        output        = toDS(pred)
        self.ds_line.setText(output)
        self.prob_line.setText(pred_proba)

        # plot probability
        plot_prob(probas=probas)
        self.prob_fig.load(sys.path[0]+'\\probability.jpg')
        self.prob_fig_label.setPixmap(self.prob_fig)
        self.prob_fig_label.setAlignment(Qt.AlignCenter)
        # self.prob_fig_label.setScaledContents(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = QtGui.QFont()
    # font.setFamily("Times New Roman") # 字体
    font.setFamily("Arial") # 字体
    font.setPointSize(15)   # 字体大小
    app.setFont(font)
    demo = predictor()
    demo.show()
    sys.exit(app.exec_())