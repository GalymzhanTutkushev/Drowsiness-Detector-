from facelib import KozTracker
import cv2
import time
import RPi.GPIO as GPIO
import sys


sys.path.insert(0,'/home/pi/Projects')
# Initialization
tracker = KozTracker()
text = "Checking..."
fps_start = time.time()
frame_num = 0
fps = 0

pin_audio = 9
pin_red = 17
pin_yellow = 27
pin_green = 11
pin_sw = 26
pin_btn = 2

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(pin_audio, GPIO.OUT, initial=0)
GPIO.setup(pin_red, GPIO.OUT, initial=0)
GPIO.setup(pin_green, GPIO.OUT, initial=0)
GPIO.setup(pin_yellow, GPIO.OUT, initial=0)
GPIO.setup(pin_btn, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(pin_sw, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

def led_control(active_pin, alarm):
    GPIO.output(active_pin, GPIO.HIGH)
    GPIO.output(pin_audio, alarm)


def led_off():
    GPIO.output(pin_red, GPIO.LOW)
    GPIO.output(pin_green, GPIO.LOW)
    GPIO.output(pin_yellow, GPIO.LOW)
    GPIO.output(pin_audio, GPIO.LOW)
    
# start main loop
while True:	
    if tracker.reset_btn_pressed:
        tracker.reset()
        led_off()

    if tracker.tracker_state == tracker.DANGER_STATE:  # no need to get frames (red zone)
        if not tracker.log_created:    
            log_text = str(tracker.koztracker_state)
            led_control(pin_red, True)  # red led
            tracker.log_event(log_text)
            tracker.video_log()
            tracker.log_created = True
    else:  # need to get frames        
        tracker.get_frame()		
        if tracker.tracker_state == tracker.NORMAL_STATE:  # green zone
            tracker.find_face()
            led_control(pin_green, False) # green led
            if tracker.face_found:
                # to be removed begin
                pt1 = (tracker.face_box.left(), tracker.face_box.top())
                pt2 = (tracker.face_box.right(), tracker.face_box.bottom())
                cv2.rectangle(tracker.current_frame, pt1, pt2, (0, 0, 0), 2)
                tracker.get_landmarks()
                for pt in tracker.current_landmarks:
                    cv2.circle(tracker.current_frame, (pt[0], pt[1]), 2, (0, 0, 0), 2)
                # to be removed end
                if tracker.both_eyes_closed():
                    if tracker.driver_state == tracker.DROWSY_STATE:
                        drowsy_time = time.time() - tracker.abnormal_state_time_start
                        drowsy_text = "%.1f" % drowsy_time
                        text = "Drowsy timer: {}".format(drowsy_text)
                        if drowsy_time > tracker.drowsy_state_timer:
                            tracker.tracker_state = tracker.ABNORMAL_STATE
                            text = "Sleeping!"
                    else:
                        tracker.driver_state = tracker.DROWSY_STATE
                        tracker.abnormal_state_time_start = time.time()
                else:
                    tracker.driver_state = tracker.NORMAL_STATE
                    normal_time = time.time() - tracker.normal_state_timer_start
                    if normal_time > tracker.normal_state_timer:
                        tracker.log_event("Плановая запись. Состояние нормальное")
                        tracker.normal_state_timer_start = time.time()
                    if tracker.blinked():
                        tracker.abnormal_state_time_start = time.time()
                    else:
                        suspicious_time = time.time() - tracker.abnormal_state_time_start
                        normal_text = "%.1f" % suspicious_time
                        text = "Normal state. Blinkless timer: {}".format(normal_text)
                        if suspicious_time > tracker.suspicious_state_timer:
                            tracker.driver_state = tracker.SUSPICIOUS_STATE
                            tracker.tracker_state = tracker.ABNORMAL_STATE
            else:  # face not found
                if tracker.driver_state == tracker.NOFACE_STATE:
                    noface_time = time.time() - tracker.abnormal_state_time_start
                    noface_text = "%.1f" % noface_time
                    text = "No face timer: {}".format(noface_text)
                    if noface_time > tracker.no_face_state_timer:
                        tracker.tracker_state = tracker.ABNORMAL_STATE
                else:
                    tracker.driver_state = tracker.NOFACE_STATE
                    tracker.abnormal_state_time_start = time.time()
        elif tracker.tracker_state == tracker.ABNORMAL_STATE:
            if tracker.driver_state == tracker.ABNORMAL_STATE:
                danger_timer = time.time() - tracker.abnormal_state_time_start
                led_control(pin_yellow, True)   # yellow led
                time.sleep(0.1)
                led_off()
                tracker.frames_to_video.append(tracker.current_frame)
                if danger_timer > tracker.danger_state_timer:
                    tracker.tracker_state = tracker.DANGER_STATE
            else:
                tracker.abnormal_state_time_start = time.time()
                tracker.koztracker_state = tracker.driver_state
                tracker.driver_state = tracker.ABNORMAL_STATE
        elif tracker.tracker_state == tracker.IDENTIFICATION_STATE:
            tracker.identify()
        frame_num += 1
        if frame_num > 10:            
            fps_time = time.time() - fps_start            
            fps = frame_num / fps_time
            frame_num = 0
            fps_start = time.time()
        fps_text = "FPS: %.1f" % fps
        cv2.putText(tracker.current_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                    lineType=cv2.LINE_AA)
        lear_text = "Left EAR: %.2f" % tracker.leye_ear_avg
        rear_text = "Right EAR: %.2f" % tracker.reye_ear_avg
        ear_text = lear_text + " " + rear_text
        cv2.putText(tracker.current_frame, ear_text, (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                    lineType=cv2.LINE_AA)
        lear_thresh_text = "Left EAR Thresh: %.2f" % tracker.driver_leye_ear_thresh
        rear_thresh_text = "Right EAR Thresh: %.2f" % tracker.driver_reye_ear_thresh
        ear_thresh_text = lear_thresh_text + " " + rear_thresh_text
        cv2.putText(tracker.current_frame, ear_thresh_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                    lineType=cv2.LINE_AA)
        cv2.putText(tracker.current_frame, text, (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                    lineType=cv2.LINE_AA)
        cv2.imshow("Frame", tracker.current_frame)

    key = cv2.waitKey(1)
    if key == ord('r'):
        tracker.reset_btn_pressed = True
    elif key & 0xFF == ord('q'):
        tracker.__del__()
        break

cv2.destroyAllWindows()
GPIO.cleanup()