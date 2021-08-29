from facelib import KozTracker
import cv2
import time


tracker = KozTracker()
faces = tracker.detector(tracker.reduced_frame, 0)

frame_num = 0
start = time.time()
fps_text = "FPS: "
while True:
    frame_num += 1
    tracker.get_frame()

    if tracker.face_found:
        pt1 = (tracker.face_box.left(), tracker.face_box.top())
        pt2 = (tracker.face_box.right(), tracker.face_box.bottom())
        tracker.get_landmarks()
        for pt in tracker.current_landmarks:
            cv2.circle(tracker.current_frame, (pt[0], pt[1]), 2, (0, 0, 0), 2)
        ear = "%.2f" % ((tracker.leye_ear + tracker.reye_ear)/2)
        text = "EAR: {}".format(ear)
        cv2.putText(tracker.current_frame, text, (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
                    lineType=cv2.LINE_AA)
        cv2.rectangle(tracker.current_frame, pt1, pt2, (0, 0, 255), 2)

    if frame_num > 60:
        fps = frame_num / (time.time() - start)
        fps = "%.2f" % fps
        fps_text = "FPS: {}".format(fps)
        frame_num = 0
        start = time.time()
    cv2.putText(tracker.current_frame, fps_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)
    text = "State: {}".format(tracker.abnormal_state_type)
    cv2.putText(tracker.current_frame, text, (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)
    if tracker.abnormal_state_type != tracker.normal_state:
        text = "Timer: {}".format(tracker.abnormal_state_time_current)
        cv2.putText(tracker.current_frame, text, (700, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
                    lineType=cv2.LINE_AA)
    cv2.imshow("Frame", tracker.current_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
