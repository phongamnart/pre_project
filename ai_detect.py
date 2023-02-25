import cv2
import time
import numpy as np
import pyrealsense2 as rs

# pipeline= rs.pipeline()
# config= rs.config()
#
# rs_w=640
# rs_h=480
# fps=60
#
# config.enable_stream(rs.stream.depth, rs_w, rs_h, rs.format.z16, fps)
# config.enable_stream(rs.stream.color, rs_w, rs_h, rs.format.bgr8, fps)
# pipeline.start(config)

cap = cv2.VideoCapture(0)
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    # frames = pipeline.wait_for_frames()
    # depth_frame = frames.get_depth_frame()
    # color_frame = frames.get_color_frame()
    #
    # depth_image = np.asanyarray(depth_frame.get_data())
    # color_image = np.asanyarray(color_frame.get_data())

    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2) #BGR
        #z = frame.get_distance(int(x), int(y))
        #print("x: ", x)
        print("-----------------------------")
        print("y: ", y)
        #print("z: ", z)


    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
