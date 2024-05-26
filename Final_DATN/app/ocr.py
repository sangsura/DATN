import cv2
def sort_contours(cnts, plate_center):
    # Sắp xếp các bounding box theo tọa độ y
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    boundingBoxes = sorted(boundingBoxes, key=lambda b: b[1])

    # Sắp xếp các bounding box vào hàng trên hoặc hàng dưới dựa vào vị trí của tâm
    top_contours = []
    bottom_contours = []
    for box in boundingBoxes:
        box_center_y = box[1] + box[3] / 2
        if box_center_y <= plate_center:
            top_contours.append(box)
        else:
            bottom_contours.append(box)

    # Sắp xếp các bounding box trong hàng trên từ trái qua phải, và trong hàng dưới từ trái qua phải
    top_contours = sorted(top_contours, key=lambda b: b[0])
    bottom_contours = sorted(bottom_contours, key=lambda b: b[0])

    # Tạo danh sách chứa các bounding box đã sắp xếp
    sorted_contours = [box for box in top_contours]
    sorted_contours.extend(bottom_contours)

    return sorted_contours
char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ'
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString