import csv
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import numpy as np
from scipy.spatial.distance import euclidean
#from timeit import default_timer as timer
import cv2
import dlib
#import face_recognition


class EyeCatcher:
    def __init__(self):
        # static variables
        self.camera = VideoStream(src=0).start()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("face_landmarks.dat")
        self.extractor = dlib.face_recognition_model_v1(
            "face_features_extractor.dat")
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart,
         self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.nose_pt = 27
        self.chin_pt = 8
        self.alpha = 0.02
        self.eye_size = 100
        self.personNames = []
        self.personFeaturesVector = []
        self.current_personFeatVector = np.zeros(128, dtype=np.float32)
        self.dbName = "faces_db.csv"
        self.readDB()

        # dynamic variables
        self.left_EAR = 0.25
        self.riht_EAR = 0.25
        self.left_EAR_AVG = 0.25
        self.riht_EAR_AVG = 0.25
        self.left_eye_frame = np.ndarray(
            shape=(self.eye_size, self.eye_size), dtype=np.uint8)
        self.riht_eye_frame = np.ndarray(
            shape=(self.eye_size, self.eye_size), dtype=np.uint8)
        self.landmarks = None
        self.blinkless_frame_counter = 0
        self.no_face_frame_counter = 0
        self.identification_frame_num = 0
        self.identification_counter = 30
        self.feature_extraction_num = 0
        self.feature_extraction_counter = 100
        self.feature_extraction_counter_open = 50
        self.feature_extraction_vec = [0] * 128
        self.blink_thresh = 0.22
        self.left_ear_open_db = 0
        self.riht_ear_open_db = 0
        self.left_ear_closed_db = 0
        self.riht_ear_closed_db = 0

        # person_state = 0 - normal
        # person_state = 1 - no blink in the last 100 frames
        # person_state = 2 - unknown person
        # person_state = 4 - sleeping
        # person_state = 8 - no face found
        # person_state = 16 - identification state
        # person_state = 32 - feature extraction state
        # person_state = 64 - Warning
        # person state = 128 - Danger
        self.person_state = 16
        self.current_number_of_faces = 0
        self.current_person_name = "Unknown"
        self.current_person_perc = 0
        self.unusual_state_counter = 0
        self.current_face = None

    def readDB(self):
        self.personNames = []
        self.personFeaturesVector = []
        with open(self.dbName, 'r') as dbFile:
            reader = csv.reader(dbFile, delimiter='|')
            for row in reader:
                self.personNames.append(row[0])
                vals = row[5].split(',')
                vals[0] = vals[0][1:]
                vals[len(vals) - 1] = vals[len(vals) - 1][:-1]
                feat_vec = []
                for val in vals:
                    feat_vec.append(float(val))
                self.personFeaturesVector.append(
                    np.asarray(feat_vec, dtype=np.float32))

    def appendDB(self, name):
        with open(self.dbName, 'a') as dbFile:
            writer = csv.writer(dbFile, delimiter='|')
            db_row = [name, self.left_ear_open_db, self.riht_ear_open_db, 
                            self.left_ear_closed_db, self.riht_ear_closed_db, self.feature_extraction_vec]
            writer.writerow(db_row)
        self.readDB()

    def createNewFeatures(self, name):
        frame = self.get_frame()
        if self.feature_extraction_num < self.feature_extraction_counter_open:
            faces = self.get_faces(frame)
            driver_face = None
            box_size = 0
            if len(faces) != 0:
                self.feature_extraction_num += 1
                for face in faces:
                    pt1 = (face.left(), face.top())
                    pt2 = (face.right(), face.bottom())
                    cur_box_size = euclidean(pt1, pt2)
                    if cur_box_size > box_size:
                        box_size = cur_box_size
                        driver_face = face
                self.analyze_face(frame, driver_face)
                self.left_ear_open_db += self.left_EAR
                self.riht_ear_open_db += self.riht_EAR
                self.current_face = driver_face
                pt1 = (self.current_face.left(), self.current_face.top())
                pt2 = (self.current_face.right(), self.current_face.bottom())
                cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
                features = self.extractor.compute_face_descriptor(frame, self.landmarks)
                for i in range(0, len(features)):
                    self.feature_extraction_vec[i] += features[i]
        elif self.feature_extraction_num < self.feature_extraction_counter:
            faces = self.get_faces(frame)
            driver_face = None
            box_size = 0
            if len(faces) != 0:
                self.feature_extraction_num += 1
                for face in faces:
                    pt1 = (face.left(), face.top())
                    pt2 = (face.right(), face.bottom())
                    cur_box_size = euclidean(pt1, pt2)
                    if cur_box_size > box_size:
                        box_size = cur_box_size
                        driver_face = face
                self.analyze_face(frame, driver_face)
                self.left_ear_closed_db += self.left_EAR
                self.riht_ear_closed_db += self.riht_EAR
                self.current_face = driver_face
                pt1 = (self.current_face.left(), self.current_face.top())
                pt2 = (self.current_face.right(), self.current_face.bottom())
                cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
                features = self.extractor.compute_face_descriptor(frame, self.landmarks)
                for i in range(0, len(features)):
                    self.feature_extraction_vec[i] += features[i]
        else:
            self.feature_extraction_num = 0
            self.left_ear_open_db /= self.feature_extraction_counter_open
            self.riht_ear_open_db /= self.feature_extraction_counter_open
            self.left_ear_closed_db /= (self.feature_extraction_counter - self.feature_extraction_counter_open)
            self.riht_ear_closed_db /= (self.feature_extraction_counter - self.feature_extraction_counter_open)
            self.person_state = 0
            for i in range(0, len(self.feature_extraction_vec)):
                self.feature_extraction_vec[i] /= self.feature_extraction_counter
            self.appendDB(name)
            self.left_ear_open_db = 0
            self.riht_ear_open_db = 0
            self.left_ear_closed_db = 0
            self.riht_ear_closed_db = 0
        return frame

    def removeFromDB(self, index):
        if index > -1 and index < len(self.personNames):
            del self.personNames[index]
            del self.personFeaturesVector[index]
            with open(self.dbName, 'w') as dbFile:
                writer = csv.writer(dbFile, delimiter='|')
                for name, vec in zip(self.personNames, self.personFeaturesVector):
                    db_row = [name, vec]
                    writer.writerow(db_row)

    @staticmethod
    def get_ear(eye):
        width = euclidean(eye[0], eye[3])
        height = euclidean(eye[1], eye[5]) + euclidean(eye[2], eye[4])
        return height / (2 * width)

    def find_person_from_db(self):
        distances = []
        for i in range(0, len(self.personFeaturesVector)):
            dist = np.linalg.norm(
                self.current_personFeatVector - self.personFeaturesVector[i])
            distances.append(dist)
        distances = np.asarray(distances)
        min_val = distances.min()
        min_ind = distances.argmin()
        person_name = "Unknown"
        if min_val < 0.5:
            person_name = self.personNames[min_ind]
        perc = (1 - min_val) * 100
        return person_name, perc

    def identification(self, frame):
        features = np.asarray(self.extractor.compute_face_descriptor(
            frame, self.landmarks), dtype=np.float32)
        # name, perc = self.find_person_from_db(features)
        return features

    def get_frame(self):
        frame = self.camera.read()
    #    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return imutils.resize(frame, width=1024)

    def get_faces(self, frame):
        return self.detector(frame, 0)

    @staticmethod
    def crop_eye(frame, eye):
        x1, y1 = eye[0]
        x2, y2 = eye[3]
        x = x1
        s = x2 - x1
        y = int((y1 + y2 - 0.4 * s) / 2)
        eye_frame = frame[y:y + int(0.4 * s), x:x + s]
        # gaussian_eye = cv2.GaussianBlur(eye_frame, (9, 9), 10.0)
        # eye_frame = cv2.addWeighted(eye_frame, 1.5, gaussian_eye, -0.5, 0, eye_frame)
        return eye_frame

    def draw_face(self, frame):
        if self.current_face is not None:
            pt1 = (self.current_face.left(), self.current_face.top())
            pt2 = (self.current_face.right(), self.current_face.bottom())
            # pt_text = (self.current_face.left() + 20,
            #           self.current_face.bottom() + 20)
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
            # font = cv2.FONT_HERSHEY_DUPLEX
            # cv2.putText(frame, self.current_person_name, pt_text, font, 0.6, (255, 255, 255), 1)

    def analyze_face(self, frame, face):
        self.landmarks = self.predictor(frame, face)
        landmarks = face_utils.shape_to_np(self.landmarks)
        left_eye = landmarks[self.lStart:self.lEnd]
        riht_eye = landmarks[self.rStart:self.rEnd]

        if self.person_state != 16:
            leftEyeHull = cv2.convexHull(left_eye)
            rightEyeHull = cv2.convexHull(riht_eye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            pt1 = (face.left(), face.top())
            pt2 = (face.right(), face.bottom())
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)

        self.left_EAR = self.get_ear(left_eye)
        self.riht_EAR = self.get_ear(riht_eye)
        cur_avg = self.alpha * self.left_EAR + \
            (1 - self.alpha) * self.left_EAR_AVG
        self.left_EAR_AVG = cur_avg
        cur_avg = self.alpha * self.riht_EAR + \
            (1 - self.alpha) * self.riht_EAR_AVG
        self.riht_EAR_AVG = cur_avg

        local_ear = (self.left_EAR + self.riht_EAR) / 2
        if local_ear > self.blink_thresh:
            self.blinkless_frame_counter += 1
        else:
            self.blinkless_frame_counter = 0

    # self.left_eye_frame = cv2.resize(self.crop_eye(frame, left_eye), (self.eye_size, self.eye_size))
    # self.riht_eye_frame = cv2.resize(self.crop_eye(frame, riht_eye), (self.eye_size, self.eye_size))

        # angle = 180 * np.arctan2(np.abs(vert_top[0]-vert_bot[0]),np.abs(vert_top[1]-vert_bot[1])) / np.pi
        # return name, perc

    def analyze_frame(self, reset, auto_id):
        frame = self.get_frame()
        faces = self.get_faces(frame)
        driver_face = None
        box_size = 0
        if self.person_state == 0:
            if len(faces) == 0:
                self.no_face_frame_counter += 1
                self.unusual_state_counter += 1
                self.blinkless_frame_counter = 0
                if self.no_face_frame_counter > 30:
                    self.person_state = 8  # no face found
            else:
                self.no_face_frame_counter = 0
                for face in faces:
                    pt1 = (face.left(), face.top())
                    pt2 = (face.right(), face.bottom())
                    cur_box_size = euclidean(pt1, pt2)
                    if cur_box_size > box_size:
                        box_size = cur_box_size
                        driver_face = face
                self.current_face = driver_face
                self.analyze_face(frame, driver_face)
                if self.current_person_name == "Unknown":
                    self.person_state = 2  # unknown person
                if self.blinkless_frame_counter > 150:
                    self.person_state = 1  # person not blinking
                if (self.left_EAR_AVG + self.riht_EAR_AVG) / 2 < self.blink_thresh:
                    self.person_state = 4  # person sleeping
        else:
            self.unusual_state_counter += 1
            if self.unusual_state_counter > 100:
                self.person_state = 64
            if self.unusual_state_counter > 200:
                self.person_state = 128

        if reset:
            if auto_id:
                self.person_state = 16
            else:
                self.person_state = 0
            self.left_EAR_AVG = 0.25
            self.riht_EAR_AVG = 0.25
            self.unusual_state_counter = 0
            self.blinkless_frame_counter = 0
            self.identification_frame_num = 0
            self.no_face_frame_counter = 0

        return frame

    def identification_state(self):
        frame = self.get_frame()
        if self.identification_frame_num < self.identification_counter:
            faces = self.get_faces(frame)
            driver_face = None
            box_size = 0
            if len(faces) != 0:
                self.identification_frame_num += 1
                for face in faces:
                    pt1 = (face.left(), face.top())
                    pt2 = (face.right(), face.bottom())
                    cur_box_size = euclidean(pt1, pt2)
                    if cur_box_size > box_size:
                        box_size = cur_box_size
                        driver_face = face
                self.analyze_face(frame, driver_face)
                self.current_face = driver_face
                pt1 = (self.current_face.left(),  self.current_face.top())
                pt2 = (self.current_face.right(), self.current_face.bottom())
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
                features = self.identification(frame)
                for i in range(0, len(features)):
                    self.current_personFeatVector[i] += features[i]
        else:
            self.identification_frame_num = 0
            for i in range(0, len(self.current_personFeatVector)):
                self.current_personFeatVector[i] /= self.identification_counter
            self.current_person_name, self.current_person_perc = self.find_person_from_db()
            self.current_personFeatVector = np.zeros(128, dtype=np.float32)
            if self.current_person_name == "Unknown":
                self.person_state = 2
            else:
                self.person_state = 0

        return frame

    def __del__(self):
        self.camera.stop()
