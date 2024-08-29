import mediapipe as mp
import cv2
import time
import numpy as np

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

from utils import draw_styled_landmarks
from utils import extract_keypoints

def mediapipe_detection(image, holistic):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturacao = image[:, :, 1]  # Saturação
    hue = image[:, :, 0]  # Matiz
    value = image[:, :, 2]  # Valor (brilho)
    saturacao = cv2.add(saturacao, 0)
    saturacao = np.clip(saturacao, 0, 255)
    hue = cv2.add(hue, 0)
    hue = np.clip(hue, 0, 255)
    value = cv2.add(value, -50)
    value = np.clip(value, 0, 255)
    image[:, :, 1] = saturacao
    image[:, :, 0] = hue
    image[:, :, 2] = value
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    results = holistic.process(image)
    return image, results

def opencam(input):
    cap = cv2.VideoCapture(input)
    last_time = time.time()
    with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            current_time = time.time()
            fps = 1 / (current_time - last_time)
            last_time = current_time
            cv2.putText(frame, f"FPS: {round(fps, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            
            imagemAlt = cv2.GaussianBlur(image, (5, 5), 0.5)

            cv2.imshow('OpenCV Feed', image)
            cv2.imshow('Imagem alterada', imagemAlt)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


