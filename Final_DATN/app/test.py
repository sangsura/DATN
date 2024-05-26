from ocr import*
import cv2
img=cv2.imread('output/a1.jpg')

gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)
# Ap dung threshold de phan tach so va nen
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
roi = img.copy()
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
cv2.imwrite('output/a2.jpg',roi)
cv2.imwrite('output/a3.jpg',binary)