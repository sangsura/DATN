
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog,QTableWidgetItem,QDialog,QScrollArea
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtCore import QTimer,QThread,pyqtSignal
from PyQt5.QtCore import Qt
from utils import *
from ocr import *
import csv
import copy
import easyocr
from ultralytics import YOLO
digit_w = 30 # Kich thuoc ki t
digit_h = 60 # Kich thuoc ki tu
model_svm = cv2.ml.SVM_load('model_trained/svm.xml')
LICENSE_MODEL_DETECTION_DIR = 'model_trained/license_plate_detector.pt'
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)
blur_path='model_trained/blur_runs-48-6288.pt' 
occ_path='model_trained/occ_runs-48-6288.pt'
att_occ=AttributeDetector_resnet(occ_path)
att_blur=AttributeDetector_resnet(blur_path)
# Khởi tạo mô hình Yolov8
model = Yolov8(onnx_model='model_trained/yolov8l_lp.onnx', confidence_thres=0.5, iou_thres=0.5)
class StartWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("GUI/start1.ui", self)  # Load giao diện start.ui
        # Kết nối sự kiện nhấn nút
        self.pushButton.clicked.connect(self.open_img_window)
        self.pushButton_3.clicked.connect(self.open_webcam_window)
        self.pushButton_2.clicked.connect(self.open_video_window)

    def open_img_window(self):
        # Hiển thị cửa sổ MainWindow khi nhấn vào label1
        self.main_window = MainWindow_IMG()
        self.main_window.show()
        self.close()

    def open_webcam_window(self):
        # Hiển thị cửa sổ MainWindow khi nhấn vào label1
        self.main_window = MainWindow_WEBCAM()
        self.main_window.show()
        self.close()

    def open_video_window(self):
        # Hiển thị cửa sổ MainWindow khi nhấn vào label1
        self.main_window = MainWindow_VIDEO()
        self.main_window.show()
        self.close()

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    id_ocr_signal = pyqtSignal(str, str,str,str)
    def __init__(self, filename,table_widget,label_lp,label_lp_thresh,label_ocr):
        super().__init__()
        self.filename = filename
        self.reader = easyocr.Reader(['en'],gpu=True)
        self.running = True
        self.custom_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
        self.table_widget = table_widget
        self.label_lp = label_lp
        self.label_lp_thresh = label_lp_thresh
        self.label_ocr = label_ocr
    def run(self):
        cap = cv2.VideoCapture(self.filename)
        id=0
        while self.running:
            ret, frame = cap.read()
            if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    '''try:'''
                    license_detections = license_plate_detector(frame)[0]
                    if len(license_detections.boxes.cls.tolist()) != 0 :
                        for license_plate in license_detections.boxes.data.tolist():
                            x1, y1, x2, y2, score, class_id = license_plate
                            if score > 0.7:
                                ocr=''
                                check=''                   
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                                gray = cv2.cvtColor( license_plate_crop, cv2.COLOR_BGR2GRAY)
                                # Ap dung threshold de phan tach so va nen
                                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                                check_occ = np.argmax(att_occ(license_plate_crop))
                                check_blur=np.argmax(att_blur(license_plate_crop))
                                if check_occ==0 and check_blur==0:
                                    check='Normal'
                                    roi = license_plate_crop.copy()
                                    #cv2.imshow("Anh bien so sau threshold", binary)
                                    #cv2.waitKey()
                                    # Segment kí tự
                                    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                                    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
                                    cont, _  = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                                    for c in sort_contours(cont,roi.shape[0]/2):
                                        (x, y, w, h) = c                                    
                                        ratio = h/w
                                        if 1.5<=ratio<=4.0: # Chon cac contour dam bao ve ratio w/h
                                            if h/roi.shape[0]>=0.3: # Chon cac contour cao tu 60% bien so tro len
                                                #print(w,h)
                                                # Ve khung chu nhat quanh so
                                                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                                # Tach so va predict
                                                curr_num = thre_mor[y:y+h,x:x+w]
                                                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                                                _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                                                
                                                curr_num = np.array(curr_num,dtype=np.float32)
                                                curr_num = curr_num.reshape(-1, digit_w * digit_h)
                                                # Dua vao model SVM
                                                result = model_svm.predict(curr_num)[1]
                                                result = int(result[0, 0])
                                                if result<=9: # Neu la so thi hien thi luon
                                                    result = str(result)
                                                else: #Neu la chu thi chuyen bang ASCII
                                                    result = chr(result)
                                                ocr +=result
                                                #print(ocr)                                                
                                elif check_occ==0 and check_blur==1:
                                    check= 'Blur'
                                elif check_occ==1 and check_blur==0: 
                                    check = 'Occlusion'
                                else:
                                    check = 'Oclusion and blur'
                                cv2.putText(frame, check+ocr, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),2)
                                self.display_cut_image_and_text(license_plate_crop,binary,ocr)
                                self.id_ocr_signal.emit(str(id),str(str(int(x1))+','+str(int(y1))+','+str(int(x2))+','+str(int(y2))),check, ocr)
                                id+=1
                            '''tracks=tracker.update_tracks(detect,frame=frame)
                            for track in tracks:
                                if track.is_confirmed():
                                    ocr=''
                                    track_id = int(track.track_id)
                                    ltrb = track.to_ltrb()
                                    class_id=track.get_det_class()
                                    x1, y1, x2, y2=map(int,ltrb)
                                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY) 
                                    result = self.reader.readtext(license_plate_crop_gray,allowlist=self.custom_characters)
                                    ocr=''
                                    for detection in result:
                                            bbox = detection[0]
                                            text = detection[1]
                                            ocr+=ocr+text
                                    ocr="ID : {0} -  {1}".format(str(track_id),str(ocr))
                                    cv2.putText(frame, ocr, (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)'''
                    h, w, ch = frame.shape
                    bytes_per_line = ch * w
                    convert_to_qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.change_pixmap_signal.emit(convert_to_qt_format)
                    '''except Exception as e:
                    print("Error:", e)'''
            else:
                break
        cap.release()
    def stop_thread(self):
        self.running = False
    def display_cut_image_and_text(self, img_lp,img_lp_thresh, text):
        cut_image_lp = img_lp.copy()  # Tạo một bản sao của ảnh để đảm bảo không chia sẻ dữ liệu
        height, width, channel = cut_image_lp.shape
        bytes_per_line = 3 * width
        cut_qimage_lp = QImage(cut_image_lp.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.label_lp.setPixmap(QPixmap.fromImage(cut_qimage_lp).scaled(self.label_lp.size(), Qt.KeepAspectRatio))
        #self.label_lp.setScaledContents(True)
        cut_image_thresh = img_lp_thresh.copy()  # Tạo một bản sao của ảnh để đảm bảo không chia sẻ dữ liệu
        height_1, width_1 = cut_image_thresh.shape
        bytes_per_line_1 =  width_1
        cut_qimage_thresh = QImage(cut_image_thresh.data, width_1, height_1, bytes_per_line_1, QImage.Format_Grayscale8)
        self.label_lp_thresh.setPixmap(QPixmap.fromImage(cut_qimage_thresh).scaled(self.label_lp_thresh.size(), Qt.KeepAspectRatio))
        #self.label_lp_thresh.setScaledContents(True)
        self.label_ocr.setText(text)
    
class MainWindow_VIDEO(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("GUI/Window_videoi.ui", self)
        self.pushButton_5.clicked.connect(self.select_video)
        self.pushButton.clicked.connect(self.open_main_window)
        self.pushButton_4.clicked.connect(self.save_id_ocr)
        self.pushButton_3.clicked.connect(self.save_to_csv)
        self.added_ids = set()
       
    def select_video(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi)")
        if filename:
            self.thread = VideoThread(filename,self.tableWidget,self.label,self.label_2,self.label_3)
            self.thread.change_pixmap_signal.connect(self.update_image) 
            self.thread.id_ocr_signal.connect(self.receive_id_ocr)
            self.thread.start()
    

    def update_image(self,image):
        self.label_4.setPixmap(QPixmap.fromImage(image).scaled(self.label_4.size(), Qt.KeepAspectRatio))

    def open_main_window(self):
        self.thread.stop_thread()
        # Hiển thị cửa sổ MainWindow khi nhấn vào label1   
        self.main_window = StartWindow()
        self.main_window.show()
        self.close()

    def save_id_ocr(self):
        # Lưu thông tin ID và OCR vào danh sách và cập nhật bảng
        if hasattr(self, 'current_id') and hasattr(self, 'current_ocr')and hasattr(self, 'current_box') and hasattr(self, 'current_check'):
            if self.current_id not in self.added_ids:
                row_count = self.tableWidget.rowCount()
                self.tableWidget.insertRow(row_count)
                self.tableWidget.setItem(row_count, 0, QTableWidgetItem(str(row_count)))
                self.tableWidget.setItem(row_count, 1, QTableWidgetItem(str(self.current_box)))
                self.tableWidget.setItem(row_count, 2, QTableWidgetItem(str(self.current_check)))
                self.tableWidget.setItem(row_count, 3, QTableWidgetItem(str(self.current_ocr)))
                self.added_ids.add(self.current_id)

    def receive_id_ocr(self, id_value,box_value,check_value, ocr_value):
        # Nhận ID và OCR từ tín hiệu và lưu vào biến tạm thời
        self.current_id = id_value
        self.current_ocr = ocr_value
        self.current_box=box_value
        self.current_check=check_value

    def save_to_csv(self):
        file_path, _ = QFileDialog.getSaveFileName(self, 'Lưu file CSV', '.', 'CSV Files (*.csv)')
        if file_path:
            with open(file_path, mode='w', newline='', encoding='utf-8') as file:  # Sử dụng encoding 'utf-8'
                writer = csv.writer(file)
                # Ghi tiêu đề của các cột
                headers = [self.tableWidget.horizontalHeaderItem(col).text() for col in range(self.tableWidget.columnCount())]
                writer.writerow(headers)
                # Ghi dữ liệu từ TableWidget vào tệp CSV
                for row in range(self.tableWidget.rowCount()):
                    row_data = []
                    for col in range(self.tableWidget.columnCount()):
                        item = self.tableWidget.item(row, col)
                        if item is not None:
                            row_data.append(item.text())
                        else:
                            row_data.append("")
                    writer.writerow(row_data)

class WebcamThread(QThread):
    frame_processed = pyqtSignal(QImage)
    id_ocr_signal = pyqtSignal(str, str,str,str)
    def __init__(self, table_widget,label_lp,label_lp_thresh,label_ocr):
        super().__init__()
        self.running = True
        self.recording = False
        self.frames = []
        self.table_widget = table_widget
        self.label_lp = label_lp
        self.label_lp_thresh = label_lp_thresh
        self.label_ocr = label_ocr
        self.custom_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
        self.video_writer = None

    def run(self):
        cap = cv2.VideoCapture(0)
        id=0
        while self.running:
            ret, frame = cap.read()
            if not ret:
                    break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                license_detections = license_plate_detector(frame)[0]
                if len(license_detections.boxes.cls.tolist()) != 0:
                        for license_plate in license_detections.boxes.data.tolist():
                            x1, y1, x2, y2, score, class_id = license_plate
                            if score > 0.7:
                                ocr=''
                                check=''                   
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                                gray = cv2.cvtColor( license_plate_crop, cv2.COLOR_BGR2GRAY)
                                # Ap dung threshold de phan tach so va nen
                                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                                check_occ = np.argmax(att_occ(license_plate_crop))
                                check_blur=np.argmax(att_blur(license_plate_crop))
                                if check_occ==0 and check_blur==0:
                                    check='Normal'
                                    roi = license_plate_crop.copy()
                                    #cv2.imshow("Anh bien so sau threshold", binary)
                                    #cv2.waitKey()
                                    # Segment kí tự
                                    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                                    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
                                    cont, _  = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                                    for c in sort_contours(cont,roi.shape[0]/2):
                                        (x, y, w, h) = c                                    
                                        ratio = h/w
                                        if 1.5<=ratio<=4.0: # Chon cac contour dam bao ve ratio w/h
                                            if h/roi.shape[0]>=0.3: # Chon cac contour cao tu 60% bien so tro len
                                                #print(w,h)
                                                # Ve khung chu nhat quanh so
                                                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                                # Tach so va predict
                                                curr_num = thre_mor[y:y+h,x:x+w]
                                                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                                                _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                                                
                                                curr_num = np.array(curr_num,dtype=np.float32)
                                                curr_num = curr_num.reshape(-1, digit_w * digit_h)
                                                # Dua vao model SVM
                                                result = model_svm.predict(curr_num)[1]
                                                result = int(result[0, 0])
                                                if result<=9: # Neu la so thi hien thi luon
                                                    result = str(result)
                                                else: #Neu la chu thi chuyen bang ASCII
                                                    result = chr(result)
                                                ocr +=result
                                                #print(ocr)                                                
                                elif check_occ==0 and check_blur==1:
                                    check= 'Blur'
                                elif check_occ==1 and check_blur==0: 
                                    check = 'Occlusion'
                                else:
                                    check = 'Oclusion and blur'
                                cv2.putText(frame, check+ocr, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),2)
                                self.display_cut_image_and_text(license_plate_crop,binary,ocr)
                                self.id_ocr_signal.emit(str(id),str(str(int(x1))+','+str(int(y1))+','+str(int(x2))+','+str(int(y2))),check, ocr)
                                id+=1
                if self.recording:
                    self.frames.append(frame)
                qimage = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                self.frame_processed.emit(qimage)
            except Exception as e:
                print("Error:", e)

    def stop_thread(self):
        self.running = False

    def start_recording(self):
        self.recording = True
        self.frames = []

    def stop_recording(self, output_path):
        self.recording = False
        if self.frames:
            height, width, _ = self.frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_path, fourcc, 20, (width, height))
            for frame in self.frames:
                self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.video_writer.release()

    def display_cut_image_and_text(self, img_lp,img_lp_thresh, text):
        cut_image_lp = img_lp.copy()  # Tạo một bản sao của ảnh để đảm bảo không chia sẻ dữ liệu
        height, width, channel = cut_image_lp.shape
        bytes_per_line = 3 * width
        cut_qimage_lp = QImage(cut_image_lp.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.label_lp.setPixmap(QPixmap.fromImage(cut_qimage_lp).scaled(self.label_lp.size(), Qt.KeepAspectRatio))
        #self.label_lp.setScaledContents(True)
        cut_image_thresh = img_lp_thresh.copy()  # Tạo một bản sao của ảnh để đảm bảo không chia sẻ dữ liệu
        height_1, width_1 = cut_image_thresh.shape
        bytes_per_line_1 =  width_1
        cut_qimage_thresh = QImage(cut_image_thresh.data, width_1, height_1, bytes_per_line_1, QImage.Format_Grayscale8)
        self.label_lp_thresh.setPixmap(QPixmap.fromImage(cut_qimage_thresh).scaled(self.label_lp_thresh.size(), Qt.KeepAspectRatio))
        #self.label_lp_thresh.setScaledContents(True)
        self.label_ocr.setText(text)
        
class MainWindow_WEBCAM(QWidget):
    def __init__(self):
        super().__init__()
        loadUi("GUI/Window_webcam.ui", self)
        self.pushButton_5.clicked.connect(self.start_webcam)
        self.pushButton.clicked.connect(self.open_main_window)
        self.pushButton_3.clicked.connect(self.save_to_csv)
        self.pushButton_2.clicked.connect(self.toggle_recording)
        self.webcam_thread = WebcamThread(self.tableWidget,self.label,self.label_2,self.label_3)
        self.webcam_thread.frame_processed.connect(self.display_webcam)
        self.webcam_thread.id_ocr_signal.connect(self.receive_id_ocr)
        self.added_ids = set()
        self.pushButton_4.clicked.connect(self.save_id_ocr)
        self.recording = False
    def open_main_window(self):
        self.webcam_thread.stop_thread()
        # Hiển thị cửa sổ MainWindow khi nhấn vào label1
        self.main_window = StartWindow()
        self.main_window.show()
        self.close()

    def start_webcam(self):
        self.webcam_thread.start()

    def display_webcam(self, qimage):
        pixmap = QPixmap.fromImage(qimage)
        self.label_4.setPixmap(pixmap.scaled(self.label_4.size(), Qt.KeepAspectRatio))

    def save_id_ocr(self):
        # Lưu thông tin ID và OCR vào danh sách và cập nhật bảng
        if hasattr(self, 'current_id') and hasattr(self, 'current_ocr')and hasattr(self, 'current_box') and hasattr(self, 'current_check'):
            if self.current_id not in self.added_ids:
                row_count = self.tableWidget.rowCount()
                self.tableWidget.insertRow(row_count)
                self.tableWidget.setItem(row_count, 0, QTableWidgetItem(str(row_count)))
                self.tableWidget.setItem(row_count, 1, QTableWidgetItem(str(self.current_box)))
                self.tableWidget.setItem(row_count, 2, QTableWidgetItem(str(self.current_check)))
                self.tableWidget.setItem(row_count, 3, QTableWidgetItem(str(self.current_ocr)))
                self.added_ids.add(self.current_id)

    def receive_id_ocr(self, id_value,box_value,check_value, ocr_value):
        # Nhận ID và OCR từ tín hiệu và lưu vào biến tạm thời
        self.current_id = id_value
        self.current_ocr = ocr_value
        self.current_box=box_value
        self.current_check=check_value

    def save_to_csv(self):
        file_path, _ = QFileDialog.getSaveFileName(self, 'Lưu file CSV', '.', 'CSV Files (*.csv)')
        if file_path:
            with open(file_path, mode='w', newline='', encoding='utf-8') as file:  # Sử dụng encoding 'utf-8'
                writer = csv.writer(file)
                # Ghi tiêu đề của các cột
                headers = [self.tableWidget.horizontalHeaderItem(col).text() for col in range(self.tableWidget.columnCount())]
                writer.writerow(headers)
                # Ghi dữ liệu từ TableWidget vào tệp CSV
                for row in range(self.tableWidget.rowCount()):
                    row_data = []
                    for col in range(self.tableWidget.columnCount()):
                        item = self.tableWidget.item(row, col)
                        if item is not None:
                            row_data.append(item.text())
                        else:
                            row_data.append("")
                    writer.writerow(row_data)
    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.recording = True
        self.pushButton_2.setText("Dừng ghi")
        self.label_4.setStyleSheet("border: 2px solid yellow;")
        self.webcam_thread.start_recording()

    def stop_recording(self):
        self.recording = False
        self.pushButton_2.setText("Ghi hình")
        self.label_4.setStyleSheet("")
        output_path, _ = QFileDialog.getSaveFileName(self, "Lưu video", "", "Video Files (*.mp4)")
        if output_path:
            self.webcam_thread.stop_recording(output_path)

'''class MainWindow_WEBCAM(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("GUI/webcam.ui", self)
        self.pushButton.clicked.connect(self.start_webcam)
        self.pushButton_2.clicked.connect(self.open_main_window)
        self.pushButton_3.clicked.connect(self.save_to_csv)
        self.webcam_thread = WebcamThread(self.tableWidget,self.label_7,self.label_8)
        self.webcam_thread.frame_processed.connect(self.display_webcam)
        self.webcam_thread.id_ocr_signal.connect(self.receive_id_ocr)
        self.added_ids = set()
        self.pushButton_4.clicked.connect(self.save_id_ocr)
        
    def open_main_window(self):
        self.webcam_thread.stop_thread()
        # Hiển thị cửa sổ MainWindow khi nhấn vào label1
        self.main_window = StartWindow()
        self.main_window.show()
        self.close()
    def start_webcam(self):
        self.webcam_thread.start()

    def display_webcam(self, qimage):
        pixmap = QPixmap.fromImage(qimage)
        self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio))

    def save_id_ocr(self):
        # Lưu thông tin ID và OCR vào danh sách và cập nhật bảng
        if hasattr(self, 'current_id') and hasattr(self, 'current_ocr'):
            if self.current_id not in self.added_ids:
                row_count = self.tableWidget.rowCount()
                self.tableWidget.insertRow(row_count)
                self.tableWidget.setItem(row_count, 0, QTableWidgetItem(str(row_count)))
                self.tableWidget.setItem(row_count, 1, QTableWidgetItem(str(self.current_ocr)))
                self.added_ids.add(self.current_id)

    def receive_id_ocr(self, id_value, ocr_value):
        # Nhận ID và OCR từ tín hiệu và lưu vào biến tạm thời
        self.current_id = id_value
        self.current_ocr = ocr_value

    def save_to_csv(self):
        file_path, _ = QFileDialog.getSaveFileName(self, 'Lưu file CSV', '.', 'CSV Files (*.csv)')
        if file_path:
            with open(file_path, mode='w', newline='', encoding='utf-8') as file:  # Sử dụng encoding 'utf-8'
                writer = csv.writer(file)
                # Ghi tiêu đề của các cột
                headers = [self.tableWidget.horizontalHeaderItem(col).text() for col in range(self.tableWidget.columnCount())]
                writer.writerow(headers)
                # Ghi dữ liệu từ TableWidget vào tệp CSV
                for row in range(self.tableWidget.rowCount()):
                    row_data = []
                    for col in range(self.tableWidget.columnCount()):
                        item = self.tableWidget.item(row, col)
                        if item is not None:
                            row_data.append(item.text())
                        else:
                            row_data.append("")
                    writer.writerow(row_data)'''
class MainWindow_IMG(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("GUI/Window_img.ui", self)
        self.pushButton_5.clicked.connect(self.reset_data)
        self.pushButton_3.clicked.connect(self.save_to_csv)
        self.pushButton_5.clicked.connect(self.open_image)
        self.pushButton.clicked.connect(self.open_main_window)

        self.custom_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
        self.reader = easyocr.Reader(['vi'], gpu=True)
        self.warped_images = []

     # Tạo QLabel để hiển thị ảnh
        self.label_image_lp = QLabel(self)
        self.label_image_lp.setGeometry(self.label.geometry())
        self.label_image_lp.setStyleSheet("border: 3px solid rgb(255, 211, 98);")
        self.label.deleteLater()  # Xóa label cũ
        self.label_image_thresh = QLabel(self)
        self.label_image_thresh.setGeometry(self.label_2.geometry())
        self.label_image_thresh.setStyleSheet("border: 3px solid rgb(255, 211, 98);")
        self.label_2.deleteLater()  # Xóa label cũ

    def open_main_window(self):
        self.main_window = StartWindow()
        self.main_window.show()
        self.close()

    def open_image(self):
        # Mở hộp thoại chọn tệp
        filename, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")

        # Kiểm tra nếu người dùng đã chọn một tệp
        if filename:
            img = cv2.imread(filename)
            if img is not None:
                boxes_info = model(img)
                count_id=0
                self.warped_images = []
                self.thresh_images = []
                for box_info in boxes_info:
                    box = box_info['box']
                    score = box_info['score']
                    class_id = box_info['class_id']
                    key_point= box_info["keypoint"]
                    print(box)
                    license_cut=img[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
                    cv2.imwrite('a.jpg',license_cut)
                    yolo_kps = np.array(key_point)
                    yolo_kps=yolo_kps[:,:2]

                    #print(yolo_kps)
                    tmp = copy.deepcopy(yolo_kps[3])
                    yolo_kps[3]= yolo_kps[2]
                    yolo_kps[2]= tmp
                # print(yolo_kps.shape)
                    yldm = copy.deepcopy(yolo_kps)	
                    #print(yldm)
                    ratio = 0
                    x1,y1,x2,y2 = get_padding_align(yldm[1],yldm[2],ratio)
                    x0,y0,x3,y3 = get_padding_align(yldm[0],yldm[3],ratio)
                    yldm[0]= np.array([x0,y0])
                    yldm[1]= np.array([x1,y1])
                    yldm[2]= np.array([x2,y2])
                    yldm[3]= np.array([x3,y3])                           
                    # print(type(yldm))
                    ypts= np.asarray([[yldm[0][0],yldm[0][1]],[yldm[1][0],yldm[1][1]],[yldm[3][0],yldm[3][1]],[yldm[2][0],yldm[2][1]]],dtype = np.float32)
                    print(ypts)
                    warped_ = four_point_transform(img, ypts)
                    cv2.imwrite('a1.jpg',warped_)
                    gray = cv2.cvtColor( warped_, cv2.COLOR_BGR2GRAY)
                    # Ap dung threshold de phan tach so va nen
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    self.warped_images.append(warped_)
                    self.thresh_images.append(binary)
                    check_occ = np.argmax(att_occ(warped_))
                    check_blur=np.argmax(att_blur(warped_))
                    print(check_occ)
                    print(check_blur)
                    check=''
                    ocr=[]
                    if check_occ==0 and check_blur==0:
                        check = 'normal'
                        # process license plate
                        license_plate_crop_gray = cv2.cvtColor(warped_, cv2.COLOR_BGR2GRAY)
                        reader = easyocr.Reader(['en'])
                        result = reader.readtext(license_plate_crop_gray)
                        for detection in result:
                            ocr.append(detection[1])
                    elif check_occ==0 and check_blur==1:
                        check = 'blur'
                    elif check_occ==1 and check_blur==0:
                        check = 'occlusion'
                    else:
                        check = 'blur and occlusion'
                    model.draw_detections(img, box, score, class_id,check,ocr)
                    result_string = ' '.join(ocr)
                    self.tableWidget.insertRow(count_id)
                    self.tableWidget.setItem(count_id, 0, QTableWidgetItem(str(count_id)))
                    self.tableWidget.setItem(count_id, 1, QTableWidgetItem(str(box)))  # Convert int to str
                    self.tableWidget.setItem(count_id, 2, QTableWidgetItem(check))
                    self.tableWidget.setItem(count_id, 3, QTableWidgetItem(result_string))
                    count_id+=1
                    self.display_img_lp(self.warped_images,self.label_image_lp)
                    self.display_img_thresh(self.thresh_images,self.label_image_thresh)

            else:
                self.result_label.setText("Không thể đọc ảnh")
        self.display_image(img, self.label_4)

    def display_image(self, cvimage, label_widget):
        height, width, channel = cvimage.shape
        bytes_per_line = 3 * width
        qimage = QImage(cvimage.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimage)
        label_widget.setPixmap(pixmap.scaled(label_widget.size(), Qt.KeepAspectRatio))

    def display_img_lp(self, images,Qlabel):
        if len(images) == 0:
            return
        # Resize all images to the same size
        label_width = Qlabel.width()
        target_height = 100  # Fixed height for all images
        target_width = label_width // 2  # Two images per row
        resized_images = [cv2.resize(image, (target_width, target_height)) for image in images]

        # Create the combined image
        rows = (len(resized_images) + 1) // 2
        combined_height = rows * target_height
        combined_width = 2 * target_width
        combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

        for idx, image in enumerate(resized_images):
            x = (idx % 2) * target_width
            y = (idx // 2) * target_height
            combined_image[y:y + target_height, x:x + target_width] = image

        height, width, channel = combined_image.shape
        bytes_per_line = 3 * width
        qimage = QImage(combined_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimage)
        Qlabel.setPixmap(pixmap)
        Qlabel.setFixedSize(combined_width, combined_height)

    def display_img_thresh(self, images,Qlabel):
        if len(images) == 0:
            return
        # Resize all images to the same size
        label_width = Qlabel.width()
        target_height = 100  # Fixed height for all images
        target_width = label_width // 2  # Two images per row
        resized_images = [cv2.resize(image, (target_width, target_height)) for image in images]
        # Create the combined image
        rows = (len(resized_images) + 1) // 2
        combined_height = rows * target_height
        combined_width = 2 * target_width
        combined_image = np.zeros((combined_height, combined_width), dtype=np.uint8)
        for idx, image in enumerate(resized_images):
            x = (idx % 2) * target_width
            y = (idx // 2) * target_height
            combined_image[y:y + target_height, x:x + target_width] = image
        height, width = combined_image.shape
        bytes_per_line = width
        qimage = QImage(combined_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8).rgbSwapped()
        pixmap = QPixmap.fromImage(qimage)
        Qlabel.setPixmap(pixmap)
        Qlabel.setFixedSize(combined_width, combined_height)

    def reset_data(self):
        self.label_4.setText("")
        self.tableWidget.setRowCount(0)
        self.label_image_lp.clear()
        self.label_image_thresh.clear()

    def save_to_csv(self):
        file_path, _ = QFileDialog.getSaveFileName(self, 'Lưu file CSV', '.', 'CSV Files (*.csv)')
        if file_path:
            with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                headers = [self.tableWidget.horizontalHeaderItem(col).text() for col in range(self.tableWidget.columnCount())]
                writer.writerow(headers)
                for row in range(self.tableWidget.rowCount()):
                    row_data = [self.tableWidget.item(row, col).text() if self.tableWidget.item(row, col) else "" for col in range(self.tableWidget.columnCount())]
                    writer.writerow(row_data)
import sys
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StartWindow()
    window.show()
    sys.exit(app.exec_())
