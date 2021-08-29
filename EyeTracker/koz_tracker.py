import cv2
import time
import os
import sys
import numpy as np

sys.path.insert(0,'/home/face/koz')
from EyeCatcher import EyeCatcher
frames_save = list()

pin_audio=21
pin_red=26
pin_green=24
pin_yellow=1
pin_btn=14
pin_sw=17
start_time = time.time()
with open("events.txt", "w") as file:
    file.write("Начало записи. Маршрут: . Машинист:")


def write_event(text):
    current_time=time.time()
    with open("events.txt", "a") as file:
        file.write("\n"+"Прошло времени: "+str(current_time-start_time)+". Состояние:"+text)


def led_control(active_pin, alarm):
    GPIO.output(active_pin, GPIO.HIGH)
    GPIO.output(pin_audio, alarm)


def led_off():
    GPIO.output(pin_red, GPIO.LOW)
    GPIO.output(pin_green, GPIO.LOW)
    GPIO.output(pin_yellow, GPIO.LOW)
    GPIO.output(pin_audio, GPIO.LOW)
    

def danger(cnt):
    if cnt > 0 and cnt <= 10:
        led_control(pin_yellow, True)
        time.sleep(0.1)
        led_off()
        write_event("Предупреждение!")
        frames_save.append(frame)   # буфферизация фреймов
    if cnt > 10 and cnt <= 20:
        led_control(pin_red, True)
        video_name = 'output' + str(round(time.time() - start_time)) + '.avi'  # название видейофайла
        write_event("Тревога!!! Нарушение зафиксировано в файле: "+ video_name)
        (h, w) = frame.shape[:2]  # размер изображения
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # кодек
        writer = None
        outFps = 20.0  # кадры в секунду при записи видеофайла

        # check if the writer is None
        if writer is None:
            # store the image dimensions, initialzie the video writer,
            writer = cv2.VideoWriter(video_name, fourcc, outFps, (w, h), True)
        for frame_save in frames_save:
            # write the output frame to file
            writer.write(frame_save)


tracker = EyeCatcher()
curp = 0

while True:
    reset_button_pressed = False
   
    key = cv2.waitKey(1) & 0xFF
    if key == ord("r") or GPIO.input(pin_btn) == GPIO.HIGH:
        reset_button_pressed = True
        led_off()
        led_control(pin_green, False)
        time.sleep(1.0)


    elif key == ord("q") or GPIO.input(pin_sw) == GPIO.HIGH:
        break

    if tracker.person_state == 16: # identification state
        tracker.identification_state()
        upper_text = "Identifying... " + str(tracker.identification_frame_num)
        frame = tracker.get_frame()
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, upper_text, (30, 30), font, 0.6, (255, 255, 255), 1)
        cv2.imshow("Frame", frame)
    else:
        frame = tracker.analyze_frame(reset_button_pressed)
        if tracker.person_state == 0:
            #upper_text = "Normal %.2f" % tracker.current_person_perc # normal state
            upper_text = "Normal Blinkless: " + str(tracker.blinkless_frame_counter) + " No face: " + str(tracker.no_face_frame_counter)# normal state
            #tracker.draw_face(frame)
            led_control(pin_green, False)
            cur_time = round(time.time() - start_time)
            if cur_time % 600 == 0:
                cur = cur_time / 60
                if cur != curp:
                    with open("events.txt", "a") as file:
                        file.write("\nНормальное состояние. Прошло минут от начала:" + str(cur))
                curp = cur
        elif tracker.person_state == 1:
            upper_text = "Not blinking " + str(tracker.unusual_state_counter) # not blinking
            danger(tracker.unusual_state_counter)
            write_event("Попытка обмана")
        elif tracker.person_state == 2:
            upper_text = "Unidentified person " + str(tracker.unusual_state_counter) + " %.2f" % tracker.current_person_perc  # unidentified person
            #tracker.draw_face(frame)
            write_event("Не идентифицирован")
        elif tracker.person_state == 4:
            upper_text = "Person sleeping " + str(tracker.unusual_state_counter) # person sleeping
            danger(tracker.unusual_state_counter)
            write_event("Нарушение бодрости")
        else:
            upper_text = "No face found " + str(tracker.unusual_state_counter) # no face found
            danger(tracker.unusual_state_counter)
            write_event("Не найдено лицо")
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, upper_text, (30, 30), font, 0.6, (255, 255, 255), 1)
        cv2.imshow("Frame", frame)
        
write_event("Сеанс работы САК завершен!")

cv2.destroyAllWindows()
writer.release()

