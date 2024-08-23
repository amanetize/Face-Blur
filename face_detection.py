from ultralytics import YOLO
import cv2
model = YOLO("yolov8n-face.pt")

img_path = "/Users/amanetize/Documents/Innohacks/input_image.jpg"

results = model(img_path)
boxes = results[0].boxes
img = cv2.imread(img_path)

for box in boxes:
    top_left_x = int(box.xyxy.tolist()[0][0])
    top_left_y = int(box.xyxy.tolist()[0][1])
    bottom_right_x = int(box.xyxy.tolist()[0][2])
    bottom_right_y = int(box.xyxy.tolist()[0][3])

    cv2.rectangle(img,(top_left_x,top_left_y), (bottom_right_x,bottom_right_y), (50,200,129),2)

    cv2.imwrite("testing.jpg", img)