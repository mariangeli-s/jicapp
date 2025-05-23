import numpy as np
import mediapipe as mp
import cv2

#para las funciones necesarias

#variable para alojar el modelo de mediapipe holistic
mp_holistic = mp.solutions.holistic
#variable para alojar el modelo de mp drawing
mp_drawing=mp.solutions.drawing_utils

#mediapipe detection
def mediapipe_detection (image,model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results
#hasta aqui

def draw_changed_landmarks(image,results):
    #Para dibujar los landmarks de las caras
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, mp_drawing.DrawingSpec(color=(191,95,0),thickness=1,circle_radius=1), mp_drawing.DrawingSpec(color=(237,178,101),thickness=1,circle_radius=0.5))
    #Dibujar landmarks de la postura
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(3,56,172), thickness=2,circle_radius=3), mp_drawing.DrawingSpec(color=(141,241,244),thickness=2,circle_radius=3))
    #Dibujar landmarks de left Hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(3,89,17),thickness=2,circle_radius=3), mp_drawing.DrawingSpec(color=(134,242,196),thickness=3,circle_radius=2))
    # Para dibujar los landmarks de la mano derecha (con sus conexiones) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(134,242,196), thickness=2,circle_radius=3), mp_drawing.DrawingSpec(color=(3,89,17), thickness=3, circle_radius=2))
    
def extract_keypoints(results): 
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in 
        results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4) 
    face = np.array([[res.x, res.y, res.z] for res in 
        results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)  
    lh = np.array([[res.x, res.y, res.z] for res in 
        results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)  
    rh = np.array([[res.x, res.y, res.z] for res in 
        results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)  
    return np.concatenate([pose, face, lh, rh])