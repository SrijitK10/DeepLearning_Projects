import cv2
from ultralytics import YOLO
from collections import defaultdict

model = YOLO("yolo11l.pt")

class_list=model.names

cap =cv2.VideoCapture('test_video/Test Video 4.mp4')

line_y_pred = 430
class_counts = defaultdict(int)
crossed_ids= set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame,persist=True, classes =[1,2,3,5,6,7])
    # print(results) 
    if results[0].boxes.data is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu()

        cv2.line(frame, (690, line_y_pred), (1130, line_y_pred),(0,0,255),3)
        cv2.putText(frame, 'Red Line', (690, line_y_pred -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1,cv2.LINE_AA)


    for box,track_id, class_idx, conf in zip(boxes,track_ids,class_indices,confidences):
        x1, y1, x2, y2 = map(int, box)

        cx = (x1+x2) //2
        cy = (y1+y2) //2


        class_name = class_list[class_idx]
        
        cv2.circle(frame, (cx,cy), 4, (0,0,255), -1)


        cv2.putText(frame, f'ID: {track_id} {class_name}',(x1,y1 -10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2)
        cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)

        if cy > line_y_pred and track_id not in crossed_ids:

            crossed_ids.add(track_id)
            class_counts[class_name]+=1

        
        y_offset = 30
        for class_name, count in class_counts.items():
            cv2.putText(frame, f"{class_name}:{count}", (50,y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
            y_offset+=30




    cv2.imshow("YOLO Object Tracking & Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cap.destroyAllWindows()