import os
import sys
import psutil
import pathlib
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy, QWidget, QVBoxLayout, QMainWindow, QApplication
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from flat import img_read, save_csv, get_img_stats
from multiprocessing import freeze_support, Process, Queue
from queue import Empty


def img_path_generator():
    for folder in AllChannels:
        img_folder = SourceFolder / folder
        if img_folder.exists():
            for root, dirs, files in os.walk(img_folder):
                for name in files:
                    name_l = name.lower()
                    if name_l.endswith(".raw") or name_l.endswith(".tif") or name_l.endswith(".tiff"):
                        img_path = os.path.join(root, name)
                        yield folder, img_path


class MultiProcessImageProcessing(Process):
    def __init__(self, queue, img_path):
        Process.__init__(self)
        self.queue = queue
        self.img_path = img_path

    def run(self):
        img_mem_map = None
        img_stats = None
        try:
            img_mem_map = img_read(self.img_path)
            if img_mem_map is not None:
                img_stats = get_img_stats(img_mem_map)
        except Exception as inst:
            print(f'Process failed for {self.img_path}.')
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .args
            print(inst)
        self.queue.put((img_mem_map, img_stats))


class MplCanvas(FigureCanvas):
    """Class to represent the FigureCanvas widget"""
    def __init__(self):
        # setup Matplotlib Figure and Axis
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        # initialization of the canvas
        FigureCanvas.__init__(self, self.fig)
        # we define the widget as expandable
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        # notify the system of updated policy
        FigureCanvas.updateGeometry(self)


class MplWidgetTest(QWidget):
    """Widget defined in Qt Designer"""
    def __init__(self, parent=None):
        # initialization of Qt MainWindow widget
        QWidget.__init__(self, parent)
        # set the canvas to the Matplotlib widget
        self.canvas = MplCanvas()
        # create a NavigationToolbar
        self.ntb = NavigationToolbar(self.canvas, self)
        # create a vertical box layout
        self.vbl = QVBoxLayout()
        # add mpl widget to vertical box
        self.vbl.addWidget(self.canvas)
        # add NavigationToolBar to vertical box
        self.vbl.addWidget(self.ntb)
        # set the layout to th vertical box
        self.setLayout(self.vbl)


class UiMplMainWindow(object):
    def __init__(self):
        self.central_widget = None
        self.gridLayout_2 = None
        self.mpl = None
        self.groupBox = None
        self.gridLayout = None
        self.buttonYes = None
        self.buttonNo = None
        self.buttonJump = None
        self.buttonSave = None
        self.menubar = None
        self.statusbar = None

    def setup_ui(self, mpl_main_window):
        mpl_main_window.setObjectName("MplMainWindow")
        mpl_main_window.resize(628, 416)
        self.central_widget = QtWidgets.QWidget(mpl_main_window)
        self.central_widget.setObjectName("central_widget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.central_widget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.mpl = MplWidgetTest(self.central_widget)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(self.mpl.sizePolicy().hasHeightForWidth())
        self.mpl.setSizePolicy(size_policy)
        self.mpl.setObjectName("mpl")
        self.gridLayout_2.addWidget(self.mpl, 0, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(self.central_widget)
        self.groupBox.setMaximumSize(QtCore.QSize(95, 16777215))
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.buttonYes = QtWidgets.QPushButton(self.groupBox)

        self.buttonYes.setMaximumSize(QtCore.QSize(75, 16777215))
        self.buttonYes.setObjectName("buttonYes")
        self.gridLayout.addWidget(self.buttonYes, 0, 0, 1, 1)

        self.buttonNo = QtWidgets.QPushButton(self.groupBox)
        self.buttonNo.setMaximumSize(QtCore.QSize(75, 16777215))
        self.buttonNo.setObjectName("buttonNo")
        self.gridLayout.addWidget(self.buttonNo, 1, 0, 1, 1)

        self.buttonJump = QtWidgets.QPushButton(self.groupBox)
        self.buttonJump.setMaximumSize(QtCore.QSize(75, 16777215))
        self.buttonJump.setObjectName("buttonJump")
        self.gridLayout.addWidget(self.buttonJump, 3, 0, 1, 1)

        self.buttonSave = QtWidgets.QPushButton(self.groupBox)
        self.buttonSave.setMaximumSize(QtCore.QSize(75, 16777215))
        self.buttonSave.setObjectName("buttonSave")
        self.gridLayout.addWidget(self.buttonSave, 4, 0, 1, 1)

        spacer_item = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacer_item, 2, 0, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox, 0, 1, 1, 1)
        mpl_main_window.setCentralWidget(self.central_widget)
        self.menubar = QtWidgets.QMenuBar(mpl_main_window)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 628, 21))
        self.menubar.setObjectName("menubar")
        mpl_main_window.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(mpl_main_window)
        self.statusbar.setObjectName("statusbar")
        mpl_main_window.setStatusBar(self.statusbar)

        self.translate_ui(mpl_main_window)
        QtCore.QMetaObject.connectSlotsByName(mpl_main_window)

    def translate_ui(self, mpl_main_window):
        translate = QtCore.QCoreApplication.translate
        mpl_main_window.setWindowTitle(translate("MplMainWindow", "MainWindow"))
        self.groupBox.setTitle(translate("MplMainWindow", "Is Flat?"))
        self.buttonYes.setText(translate("MplMainWindow", "Yes"))
        self.buttonNo.setText(translate("MplMainWindow", "No"))
        self.buttonJump.setText(translate("MplMainWindow", "Jump 100"))
        self.buttonSave.setText(translate("MplMainWindow", "Save"))


class DesignerMainWindow(QMainWindow, UiMplMainWindow):
    def __init__(self, parent=None):
        super(DesignerMainWindow, self).__init__(parent)
        self.setup_ui(self)
        # connect the signals with the slots
        self.buttonYes.clicked.connect(self.save_yes)
        self.buttonNo.clicked.connect(self.save_no)
        self.buttonJump.clicked.connect(self.jump)
        self.buttonSave.clicked.connect(self.save_to_csv)
        self.img_stats_list = [['channel'] + get_img_stats(None) + ['path'] + ['flat']]
        self.img_mem_map = None
        self.img_stats = None
        self.yes_counter = 0
        self.no_counter = 0
        self.folder = None
        self.img_path = None
        self.img_path_generator = img_path_generator()
        self.queue = Queue()
        for _ in range(psutil.cpu_count(logical=True)):
            self.folder, self.img_path = next(self.img_path_generator)
            MultiProcessImageProcessing(self.queue, self.img_path).start()
        self.img_show_next()

    def save_yes(self):
        self.update_stats_class("yes")
        self.yes_counter += 1
        self.img_show_next()

    def save_no(self):
        self.update_stats_class("no")
        self.no_counter += 1
        self.img_show_next()

    def img_show_next(self):
        # if self.yes_counter > MinRequiredSamplePerClass and self.no_counter > MinRequiredSamplePerClass:
        #
        # else:
        could_not_get_something = True
        while could_not_get_something:
            try:
                self.img_mem_map, self.img_stats = self.queue.get(block=False)
                if self.img_mem_map is not None:
                    could_not_get_something = False
                self.folder, self.img_path = next(self.img_path_generator)
                MultiProcessImageProcessing(self.queue, self.img_path).start()
            except Empty:
                pass
        self.img_erase()
        self.mpl.canvas.ax.imshow(self.img_mem_map)
        self.mpl.canvas.ax.relim()
        self.mpl.canvas.ax.autoscale(True)
        self.mpl.ntb.update()  # <-- add this
        # self.mpl.ntb.push_current()      #     and possibly this(?)
        self.mpl.canvas.draw()

    def jump(self):
        for _ in range(100):
            next(self.img_path_generator)
        self.img_show_next()

    def img_erase(self):
        self.mpl.canvas.ax.clear()
        self.mpl.ntb.update()  # <-- add this
        # self.mpl.ntb.push_current()      #     and possibly this(?)
        self.mpl.canvas.draw()

    def update_stats_class(self, img_class):
        self.img_stats_list += [[self.folder] + self.img_stats + [self.img_path] + [img_class]]

    def save_to_csv(self):
        save_csv(SourceFolder / (self.folder + '_image_stats_classes.csv'), self.img_stats_list)


if __name__ == '__main__':
    freeze_support()
    AllChannels = ["Ex_561_Em_600", "Ex_488_Em_525", "Ex_642_Em_680", ]  #
    SourceFolder = pathlib.Path(r"F:\20210907_16_56_41_SM210705_01_LS_4X_4000z_Compressed")
    MinRequiredSamplePerClass = 1000
    app = QApplication(sys.argv)
    dmw = DesignerMainWindow()
    # show it
    dmw.show()
    sys.exit(app.exec_())
