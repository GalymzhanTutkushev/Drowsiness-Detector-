import numpy as np
import cv2
import dlib
import imutils
from imutils.video.pivideostream import PiVideoStream
from imutils import face_utils
import time
import datetime
from scipy.spatial.distance import euclidean
import csv


class KozTracker:
    def __init__(self):
        self.frame_width = 960
        self.aspect_ratio = 16 / 9                
        self.frame_height = int(self.frame_width / self.aspect_ratio)
        self.frame_height = 544
        self.camera = PiVideoStream(resolution=(self.frame_width, self.frame_height)).start()        
        time.sleep(2.0)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("face_landmarks.dat")
        self.extractor = dlib.face_recognition_model_v1("face_features_extractor.dat")
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.media_path = "/media/pi/SAQ/"
        self.dbName = self.media_path + "faces_db.csv"
        self.logName = self.media_path + "saq_registry.log"
        self.epoch_start = time.time()
        self.frames_to_video = []

        # boolean variables as state changes
        self.reset_btn_pressed = True
        self.log_created = False

        # personal parameters
        self.driver_name = "Unknown"
        self.driver_facial_features = np.zeros(128, dtype=np.float32)
        self.driver_leye_ear_thresh = 0.25
        self.driver_reye_ear_thresh = 0.25
        self.driver_leye_blink_thresh = 0.27
        self.driver_reye_blink_thresh = 0.27
        self.personNames = []
        self.personFeaturesVector = []
        self.person_open_ears = []
        self.person_closed_ears = []
        self.current_features = []

        # tuning parameters        
        self.reduction_coeff = 3     
        self.reduced_width = int(self.frame_width / self.reduction_coeff)
        self.reduced_height = int(self.frame_height / self.reduction_coeff)
        self.face_box_search_thresh = 10
        self.alpha = 0.25
        self.identification_perc_thresh = 0.7

        # timers for states (in seconds)
        self.drowsy_state_timer = 30
        self.no_face_state_timer = 30
        self.suspicious_state_timer = 180
        self.danger_state_timer = 30
        self.normal_state_timer = 600

        # dynamic variables
        self.current_frame = np.zeros((self.frame_width, self.frame_height), dtype=np.uint8)
        self.reduced_frame = np.zeros((self.reduced_width, self.reduced_height), dtype=np.uint8)
        self.leye_ear = 0
        self.reye_ear = 0
        self.leye_ear_avg = 0
        self.reye_ear_avg = 0
        self.face_box = dlib.rectangle(0, 0, 0, 0)
        self.face_found = False
        self.current_landmarks = np.zeros((68, 2), dtype=np.uint16)
        self.current_landmarks_to_id = None
        self.driver_state = 0  # driver state, can be: NORMAL, DROWSY, NOFACE, SUSPICIOUS, NOTIDENTIFIED
        self.tracker_state = 0  # system state, can be: NORMAL, ABNORMAL, DANGER, IDENTIFICATION
        self.koztracker_state = 0

        # state types
        self.NORMAL_STATE = 0
        self.ABNORMAL_STATE = 1
        self.DANGER_STATE = 2
        self.IDENTIFICATION_STATE = 3

        self.NOFACE_STATE = 4
        self.DROWSY_STATE = 5
        self.SUSPICIOUS_STATE = 6
        self.NOTIDENTIFIED_STATE = 7

        # counters for various events
        self.face_box_search_cnt = 1
        self.abnormal_state_time_start = 0
        self.normal_state_timer_start = 0
        self.abnormal_state_type = 0
        self.identification_cnt = 0
        self.identification_cnt_max = 10
        self.event_num = 0

        # initialize
        self.log_event("Program start")
        self.readDB()
        self.get_frame()

    def reset(self):
        self.reset_btn_pressed = False
        self.log_created = False
        self.tracker_state = self.NORMAL_STATE
        self.driver_state = self.NORMAL_STATE
        self.koztracker_state = self.NORMAL_STATE
        self.face_box_search_cnt = 1
        self.abnormal_state_time_start = time.time()
        self.normal_state_timer_start = time.time()
        self.abnormal_state_type = 0
        self.frames_to_video = []

    def log_event(self, event_text):        
        self.event_num += 1
        with open(self.logName, "a") as logfile:
            time_passed = int(time.time() - self.epoch_start)
            event_text = str(self.event_num) + ": " + str(datetime.timedelta(seconds=time_passed)) + " - " + event_text + "\n"
            logfile.write(event_text)

    def get_frame(self):  # get frame from camera. Each 5th frame get a smaller frame and search for a face
        frame = self.camera.read()
        if frame is not None:
            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.face_box_search_cnt -= 1
            if self.face_box_search_cnt == 0:
                self.reduced_frame = imutils.resize(self.current_frame, width=self.reduced_width)
                self.find_face()
                self.face_box_search_cnt = self.face_box_search_thresh

    def find_face(self):  # searches face in a reduced frame box and gets the largest box
        lfaces = self.detector(self.reduced_frame, 0)
        if len(lfaces) > 0:
            self.face_found = True
            null_box = dlib.rectangle(0, 0, 0, 0)
            for lface in lfaces:
                if lface.area() > null_box.area():
                    null_box = lface
            self.face_box = dlib.rectangle(int(self.reduction_coeff*null_box.left()),
                                           int(self.reduction_coeff*null_box.top()),
                                           int(self.reduction_coeff*null_box.right()),
                                           int(self.reduction_coeff*null_box.bottom()))
        else:
            self.face_found = False

    @staticmethod
    def get_ear(eye):
        width = euclidean(eye[0], eye[3])
        height = euclidean(eye[1], eye[5]) + euclidean(eye[2], eye[4])
        return height / (2 * width)

    def get_landmarks(self):  # get landmarks for a face in face_box and calculate EAR for each eye
        self.current_landmarks_to_id = self.predictor(self.current_frame, self.face_box)
        self.current_landmarks = face_utils.shape_to_np(self.current_landmarks_to_id)
        self.leye_ear = self.get_ear(self.current_landmarks[self.lStart:self.lEnd])
        self.reye_ear = self.get_ear(self.current_landmarks[self.rStart:self.rEnd])

    def both_eyes_closed(self):
        leye_cur_avg = self.alpha * self.leye_ear + (1 - self.alpha) * self.leye_ear_avg
        self.leye_ear_avg = leye_cur_avg
        reye_cur_avg = self.alpha * self.reye_ear + (1 - self.alpha) * self.reye_ear_avg
        self.reye_ear_avg = reye_cur_avg

        if leye_cur_avg < self.driver_leye_ear_thresh and reye_cur_avg < self.driver_reye_ear_thresh:
            return True
        else:
            return False

    def blinked(self):
        if self.leye_ear < self.driver_leye_ear_thresh and self.reye_ear < self.driver_reye_ear_thresh:
            return True
        else:
            return False

    def readDB(self):
        self.personNames = []
        self.personFeaturesVector = []
        self.person_open_ears = []
        self.person_closed_ears = []
        with open(self.dbName, 'r') as dbFile:
            reader = csv.reader(dbFile, delimiter='|')
            for row in reader:                
                self.personNames.append(row[0])
                left = float(row[1])
                riht = float(row[2])
                self.person_open_ears.append((left, riht))
                left = float(row[3])
                riht = float(row[4])
                self.person_closed_ears.append((left, riht))
                vals = row[5].split(',')
                vals[0] = vals[0][1:]
                vals[len(vals) - 1] = vals[len(vals) - 1][:-1]
                feat_vec = []
                for val in vals:
                    feat_vec.append(float(val))
                self.personFeaturesVector.append(
                    np.asarray(feat_vec, dtype=np.float32))
        leye_ear_max, reye_ear_max = self.person_open_ears[0]
        leye_ear_min, reye_ear_min = self.person_closed_ears[0]
        self.driver_leye_ear_thresh = leye_ear_min + 0.3 * (leye_ear_max - leye_ear_min)
        self.driver_reye_ear_thresh = reye_ear_min + 0.3 * (reye_ear_max - reye_ear_min)
        self.driver_leye_blink_thresh = leye_ear_min + 0.5 * (leye_ear_max - leye_ear_min)
        self.driver_reye_blink_thresh = reye_ear_min + 0.5 * (reye_ear_max - reye_ear_min)

    def identify(self):
        if self.identification_cnt < self.identification_cnt_max:
            self.reduced_frame = cv2.cvtColor(imutils.resize(self.current_frame, width=self.reduced_width),
                                              cv2.COLOR_BGR2GRAY)
            self.find_face()
            if self.face_found:
                self.get_landmarks()
                features = np.asarray(self.extractor.compute_face_descriptor(self.current_frame,
                                                                             self.current_landmarks_to_id),
                                      dtype=np.float32)
                self.current_features.append(features)
                self.identification_cnt += 1
            print("Identifying... - ", self.identification_cnt)
        else:
            self.identification_cnt = 0
            feats = np.asarray(self.current_features)
            feat = feats.mean(axis=0)
            name = self.find_person_from_db(feat)
            if name == "Unknown":
                self.tracker_state = self.NOTIDENTIFIED_STATE
                #self.log_event("Неопознанная личность")
            else:
                self.tracker_state = self.NORMAL_STATE
                log_text = "Личность машиниста: " + name
                #self.log_event(log_text)
            print("Name: ", self.driver_name)
            print("Left EAR Drowsiness Thresh: ", self.driver_leye_ear_thresh)
            print("Right EAR Drowsiness Thresh: ", self.driver_reye_ear_thresh)
            print("Left EAR Blink Thresh: ", self.driver_leye_blink_thresh)
            print("Right EAR Blink Thresh: ", self.driver_reye_blink_thresh)

    def find_person_from_db(self, feat):
        distances = []
        for personVector in self.personFeaturesVector:
            dist = np.linalg.norm(feat - personVector)
            distances.append(dist)
        distances = np.asarray(distances)
        min_val = distances.min()
        min_ind = distances.argmin()
        person_name = "Unknown"
        if min_val < self.identification_perc_thresh:
            person_name = self.personNames[min_ind]
            self.driver_name = person_name
            leye_ear_max, reye_ear_max = self.person_open_ears[min_ind]
            leye_ear_min, reye_ear_min = self.person_closed_ears[min_ind]
            self.driver_leye_ear_thresh = leye_ear_min + 0.3 * (leye_ear_max - leye_ear_min)
            self.driver_reye_ear_thresh = reye_ear_min + 0.3 * (reye_ear_max - reye_ear_min)
            self.driver_leye_blink_thresh = leye_ear_min + 0.5 * (leye_ear_max - leye_ear_min)
            self.driver_reye_blink_thresh = reye_ear_min + 0.5 * (reye_ear_max - reye_ear_min)
        return person_name

    def video_log(self):
        video_name = self.media_path + "video_log_" + str(self.event_num) + ".avi"
        out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 120,
                              (self.frame_width, self.frame_height))
        for frame in self.frames_to_video:
            out.write(frame)

    def __del__(self):
        self.camera.stop()
        self.log_event("Program stop")
