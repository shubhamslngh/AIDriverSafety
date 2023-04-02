import numpy as np
import cv2
import dlib
import time
from scipy.spatial import distance as dist
from imutils import face_utils
import playsound


def trigger_alarm():
    playsound.playsound('alarm2.mp3')


def cal_yawn(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = dist.euclidean(top_mean, low_mean)
    return distance


def cal_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A+B) / (2.0*C)
    return ear


cam = cv2.VideoCapture(0)

# -------Models---------#
face_model = dlib.get_frontal_face_detector()
landmark_model = dlib.shape_predictor(
    './shape_predictor_68_face_landmarks.dat')

# --------Variables-------#
yawn_thresh = 40
ear_thresh = 0.25
ptime = 0

while True:
    suc, frame = cam.read()

    if not suc:
        break

    # ---------FPS------------#
    ctime = time.time()
    fps = int(1 / (ctime - ptime))
    ptime = ctime
    cv2.putText(frame, f'FPS: {fps}', (20, 50),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 200, 255), 3)

    # ------Detecting face------#
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_model(img_gray)
    if len(faces) == 0:
        cv2.putText(frame, f'NO FACE DETECTED', (255, 250),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
    else:
        for face in faces:

            # ----------Detect Landmarks-----------#
            shapes = landmark_model(img_gray, face)
            shape = face_utils.shape_to_np(shapes)

            # -------Detecting/Marking the lower and upper lip--------#
            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 165, 255), thickness=3)

            # -------Calculating the lip distance-----#
            lip_dist = cal_yawn(shape)

            # -------Detecting/Marking eyes---------#
            eyes = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
            left_eye = shape[36:42]
            right_eye = shape[42:48]
            cv2.drawContours(frame, [left_eye], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye], -1, (0, 255, 0), 1)
            # -------Calculating Eye Aspect Ratio(EAR)-----#
            left_ear = cal_ear(left_eye)
            right_ear = cal_ear(right_eye)
            ear = (left_ear+right_ear)/2

        # -------Detecting/Marking the mouth-----------#
            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 165, 255), thickness=3)

        # -------Calculating the lip distance-----#
            lip_dist = cal_yawn(shape)

        # -------Checking for Yawn and Eyes closed-------#
            if lip_dist > yawn_thresh and ear < ear_thresh:
                cv2.putText(frame, f'User Yawning! and Eye blinking', (255, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
            elif ear < ear_thresh:
                cv2.putText(frame, f'eye blink!', (20, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            elif lip_dist > yawn_thresh:
                trigger_alarm()
                cv2.putText(frame, f'User Yawning!', (20, 500),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 0, 255), 2)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
