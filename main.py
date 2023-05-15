import sys
import cv2
import joblib
import numpy as np
from PyQt5 import uic, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from PIL import Image as ImagePil
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

#?-----------------------------------------------------------------------------------------------------------------------------#

class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        uic.loadUi(r"AdvancedProject.ui", self)
        self.Browse_pushButton.clicked.connect(self.BrowseImage) ###connections###
        self.Classify_pushButton.clicked.connect(self.classify)
        self.show()

    def BrowseImage(self):
        self.lineEdit_2.clear()
        file_name=QtWidgets.QFileDialog.getOpenFileName(self, "Browse Image", "../", "*.jpg;;" "*.png;;" )
        self.file_path =file_name[0]
        if file_name[0].endswith(".jpg") or file_name[0].endswith(".png") :
            try:
                Pil_img=ImagePil.open(self.file_path)
                global image_cv
                        #####using Pil library to open the image#####
                Pixmap_img=QPixmap(self.file_path)
                       ####using QPixmap to plot the image on a label####
                self.image_cv=plt.imread(self.file_path,0)

                self.Plottingwidget.canvas.axes.clear()
                self.Plottingwidget.canvas.axes.imshow(self.image_cv)
                self.Plottingwidget.canvas.draw()
            except:
                self.ShowPopUpMessage("image is corrupted.")

    def classify(self):
         
        resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        svm_model = joblib.load('svm_model50.pkl')
        # Load and preprocess the image
        img = image.load_img(self.file_path , target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Extract features
        resnet_features = resnet_model.predict(x)
        # Flatten the features
        resnet_features_flat = np.reshape(resnet_features, (resnet_features.shape[0], -1))
        # Predict the classes using the SVM model
        svm_pred = svm_model.predict(resnet_features_flat)

        class_names = ['Drusen', 'Exudate', 'Exudate']

        self.lineEdit_2.setText(f'{class_names[svm_pred[0]]}')

#?-----------------------------------------------------------------------------------------------------------------------------#

                                                #?######## General Helper Functions  #########

    #! Show an Error Message for Handling Invalid files
    #* Take Error message as a text
    def ShowPopUpMessage(self, popUpMessage):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Error")
        msg.setInformativeText(popUpMessage)
        msg.setWindowTitle("Error")
        msg.exec_()


#?-----------------------------------------------------------------------------------------------------------------------------#


from pyqtgraph import PlotWidget
app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()                      
        
        


            
        
 