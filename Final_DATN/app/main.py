import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog,QTableWidgetItem,QDialog
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtCore import QTimer,QThread,pyqtSignal
from PyQt5.QtCore import Qt
from utils import *
import csv
import copy
import easyocr
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
LICENSE_MODEL_DETECTION_DIR = 'model_trained/license_plate_detector.pt'
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)
blur_path='model_trained/blur_runs-48-6288.pt' 
occ_path='model_trained/occ_runs-48-6288.pt'
att_occ=AttributeDetector_resnet(occ_path)
att_blur=AttributeDetector_resnet(blur_path)
tracker=DeepSort(max_age=5)
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
    def __init__(self, filename,table_widget):
        super().__init__()
        self.filename = filename
        self.reader = easyocr.Reader(['en'])
        self.running = True
        self.table_widget = table_widget
    def run(self):
        cap = cv2.VideoCapture(self.filename)
        count_id=[]
        while self.running:
            ret, frame = cap.read()
            if ret:
                    detect=[]
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
                    license_detections = license_plate_detector(frame)[0]
                    if len(license_detections.boxes.cls.tolist()) != 0 :
                            for license_plate in license_detections.boxes.data.tolist() :
                                x1, y1, x2, y2, score, class_id = license_plate
                                if score>0.7:
                                    detect.append([[x1,y1,x2-x1,y2-y1],score,class_id])
                                '''cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY) 
                                result = self.reader.readtext(license_plate_crop_gray)
                                for detection in result:
                                    bbox = detection[0]
                                    text = detection[1]
                                    cv2.putText(frame, text, (int(bbox[0][0]), int(bbox[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)'''
                            tracks=tracker.update_tracks(detect,frame=frame)
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
                                    result = self.reader.readtext(license_plate_crop_gray)
                                    ocr=''
                                    for detection in result:
                                            bbox = detection[0]
                                            text = detection[1]
                                            ocr+=ocr+text
                                    ocr="ID : {0} -  {1}".format(str(track_id),str(ocr))
                                    cv2.putText(frame, ocr, (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                                    if track_id not in count_id:
                                        count_id.append(track_id)
                                        self.table_widget.insertRow(len(count_id)-1)
                                    self.table_widget.setItem(track_id-1, 0, QTableWidgetItem(str(track_id)))
                                    self.table_widget.setItem(track_id-1, 1, QTableWidgetItem(str(ocr)))
                                    print(track_id)
                                    print(count_id) 
                    h, w, ch = frame.shape
                    bytes_per_line = ch * w
                    convert_to_qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.change_pixmap_signal.emit(convert_to_qt_format)
                
            else:
                break
        cap.release()
    def stop_thread(self):
        self.running = False
class MainWindow_VIDEO(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("GUI/video.ui", self)
        self.pushButton.clicked.connect(self.select_video)
        self.pushButton_2.clicked.connect(self.open_main_window)
    def select_video(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi)")
        if filename:
            self.thread = VideoThread(filename,self.tableWidget)
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.start()
    def update_image(self,image):
        self.label.setPixmap(QPixmap.fromImage(image).scaled(self.label.size(), Qt.KeepAspectRatio))
    def open_main_window(self):
        self.thread.stop_thread()
        # Hiển thị cửa sổ MainWindow khi nhấn vào label1   
        self.main_window = StartWindow()
        self.main_window.show()
        self.close()
    def display_image(self, cvimage, label_widget):
        # Chuyển đổi ảnh từ OpenCV sang QImage
        height, width, channel = cvimage.shape
        bytes_per_line = 3 * width
        qimage = QImage(cvimage.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        # Hiển thị ảnh lên QLabel
        pixmap = QPixmap.fromImage(qimage)
        label_widget.setPixmap(pixmap.scaled(label_widget.size(), Qt.KeepAspectRatio))
class WebcamThread(QThread):
    frame_processed = pyqtSignal(QImage)

    def __init__(self,table_widget):
        super().__init__()
        self.reader = easyocr.Reader(['en'])
        self.running = True
        self.table_widget = table_widget
    def display_info(self, track_id, ocr):
        if track_id in self.track_info:  # Kiểm tra xem ID đã tồn tại trong dictionary chưa
            # Nếu đã tồn tại, chỉ cập nhật thông tin
            row = self.track_info[track_id]
            self.table_widget.setItem(row, 1, QTableWidgetItem(str(ocr)))
        else:
            # Nếu chưa tồn tại, thêm mới thông tin và cập nhật dictionary
            row_count = self.table_widget.rowCount()
            self.table_widget.insertRow(row_count)
            self.table_widget.setItem(row_count, 0, QTableWidgetItem(str(track_id)))
            self.table_widget.setItem(row_count, 1, QTableWidgetItem(str(ocr)))
            self.track_info[track_id] = row_count  # Lưu vị trí hàng của ID vào dictionary
    def run(self):
            cap = cv2.VideoCapture(0)
            count_id=[]
            detect=[]
            while self.running:
                ret, frame = cap.read()
                if not ret:
                        break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    license_detections = license_plate_detector(frame)[0]
                    if len(license_detections.boxes.cls.tolist()) != 0 :
                        for license_plate in license_detections.boxes.data.tolist() :
                            x1, y1, x2, y2, score, class_id = license_plate
                            if score>0.7:
                                detect.append([[x1,y1,x2-x1,y2-y1],score,class_id])        
                    tracks=tracker.update_tracks(detect,frame=frame)
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
                            result = self.reader.readtext(license_plate_crop_gray)
                            for detection in result:
                                            bbox = detection[0]
                                            text = detection[1]
                                            ocr+=ocr+text
                            ocr="ID : {0} -  {1}".format(str(track_id),str(ocr))
                            cv2.putText(frame, ocr, (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            if track_id not in count_id:
                                count_id.append(track_id)
                                self.table_widget.insertRow(len(count_id)-1)
                            self.table_widget.setItem(track_id-1, 0, QTableWidgetItem(str(track_id)))
                            self.table_widget.setItem(track_id-1, 1, QTableWidgetItem(str(ocr))) 
                    qimage = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                    self.frame_processed.emit(qimage)
                except:
                    print("loi")
    def stop_thread(self):
        self.running = False
class MainWindow_WEBCAM(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("GUI/webcam.ui", self)
        self.pushButton.clicked.connect(self.start_webcam)
        self.pushButton_2.clicked.connect(self.open_main_window)
        self.webcam_thread = WebcamThread(self.tableWidget)
        self.webcam_thread.frame_processed.connect(self.display_webcam)
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
    def display_image(self, cvimage, label_widget):
        # Chuyển đổi ảnh từ OpenCV sang QImage
        height, width, channel = cvimage.shape
        bytes_per_line = 3 * width
        qimage = QImage(cvimage.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        # Hiển thị ảnh lên QLabel
        pixmap = QPixmap.fromImage(qimage)
        label_widget.setPixmap(pixmap.scaled(label_widget.size(), Qt.KeepAspectRatio))
class MainWindow_IMG(QMainWindow):
    def __init__(self):
        super().__init__()
        # Tải giao diện từ file .ui
        loadUi("GUI/main_img.ui", self)
        self.pushButton.clicked.connect(self.reset_data)
         # Tạo nút lưu và kết nối với sự kiện nhấn
        
        self.pushButton_2.clicked.connect(self.save_to_csv)
        # Kết nối sự kiện nhấn nút pushButton với phương thức open_image
        self.pushButton.clicked.connect(self.open_image)
        self.pushButton_3.clicked.connect(self.open_main_window)
        ''' path_img_ui1='img_ui\z5312217369290_f95ca37c21ebad9381e285b6319b68ee.jpg'
        img_ui1=cv2.imread(path_img_ui1)
        self.display_image(img_ui1, self.label_5)'''

    def open_main_window(self):
        # Hiển thị cửa sổ MainWindow khi nhấn vào label1
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
                for box_info in boxes_info:
                    box = box_info['box']
                    score = box_info['score']
                    class_id = box_info['class_id']
                    key_point= box_info["keypoint"]
                    print(key_point)
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
            else:
                self.result_label.setText("Không thể đọc ảnh")
        self.display_image(img, self.label_2)
        
    def display_image(self, cvimage, label_widget):
        # Chuyển đổi ảnh từ OpenCV sang QImage
        height, width, channel = cvimage.shape
        bytes_per_line = 3 * width
        qimage = QImage(cvimage.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        # Hiển thị ảnh lên QLabel
        pixmap = QPixmap.fromImage(qimage)
        label_widget.setPixmap(pixmap.scaled(label_widget.size(), Qt.KeepAspectRatio))
    def reset_data(self):
        # Xóa văn bản của label
        self.label_2.setText("")

        # Xóa dữ liệu của bảng
        self.tableWidget.setRowCount(0)
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StartWindow()
    window.show()
    sys.exit(app.exec_())
