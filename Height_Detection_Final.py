import cv2
import math
import numpy as np

net = cv2.dnn.readNet("C:/Users/Anisha/OneDrive/Desktop/Python/OpenCV_Project.py/yolo-coco/yolov3.weights",
                      "C:/Users/Anisha/OneDrive/Desktop/Python/OpenCV_Project.py/yolo-coco/yolov3.cfg")
with open("C:/Users/Anisha/OneDrive/Desktop/Python/OpenCV_Project.py/yolo-coco/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames() 
outputlayers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()] 
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

PxW = 400
camdist = 880
RW = 550

def object_detector(frame):
    datalist = []
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(outputlayers)

    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)#
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color_index = class_ids[i]
            if color_index < len(colors):
                color = tuple(map(int, colors[color_index]))
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y), font, 3, color, 1)

                if label == 'person':
                    datalist.append([label, w, h, (x, y)])

    return datalist

# Formula: focal_length = (width in pixels * distance from camera) / real width
fl = (PxW * camdist) / RW

def distance_from_cam(focal_len, real_object_width, pixelwidth):
    dist = ((real_object_width * focal_len) / pixelwidth) / 10
    return dist

def calculate_real_height(pixelheight, real_distance, focal_distance):
    real_height = ((pixelheight * real_distance) / focal_distance) 
    return real_height

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (0, 0), fx=2, fy=2)

        data = object_detector(frame)
        for d in data:
            pixelwidth = d[1] 
            pixelheight = d[2]
            dist = distance_from_cam(fl, RW, pixelwidth)
            x, y = d[3]

            cv2.putText(frame, "dist = {:.2f}".format(dist), (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            realheight = calculate_real_height(pixelheight, dist, fl) 
            cv2.putText(frame, "RealHeight = {:.2f}".format(calculate_real_height(pixelheight, dist, fl)), (x, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Real-time Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Failed to read frame from the camera.")

cap.release()
cv2.destroyAllWindows()
