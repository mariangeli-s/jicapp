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

#NORMALIZACIÓM
#HANDBOX
def draw_hand_bbox(image, hand_landmarks):
    if hand_landmarks is None:
        return  # No hacer nada si no hay landmarks

    h, w, _ = image.shape
    coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
    
    x_vals = [pt[0] for pt in coords]
    y_vals = [pt[1] for pt in coords]
    
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)

    # Dibuja el rectángulo
    cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (255, 255, 0), 2)
    
#DIBUJAR LANDMARKS CON HAND BOX
#le cambié el nombreeeeeee _box
def draw_changed_landmarks_box(image, results): 
    # Para dibujar los landmarks de la cara (con sus conexiones) 
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, mp_drawing.DrawingSpec(color=(191,95,0), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(237,178,101), thickness=1, circle_radius=0.5))  
    
    # Para dibujar los landmarks de la postura (con sus conexiones) 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(3,56,172), thickness=2, circle_radius=3), mp_drawing.DrawingSpec(color=(141,241,244), thickness=2, circle_radius=3))  
    
    # Para dibujar los landmarks de la mano izquierda (con sus conexiones) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(3,89,17), thickness=2, circle_radius=3), mp_drawing.DrawingSpec(color=(134,242,196), thickness=3, circle_radius=2)) 

    # Dibujar bounding box
    draw_hand_bbox(image, results.left_hand_landmarks)
    
     # Para dibujar los landmarks de la mano derecha (con sus conexiones) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(134,242,196), thickness=2, circle_radius=3), mp_drawing.DrawingSpec(color=(3,89,17), thickness=3, circle_radius=2)) 

    # Dibujar bounding box
    draw_hand_bbox(image, results.right_hand_landmarks)
    
#FUNCION DE NORMALIZACIÓN
def normalize_hand_landmarks(hand_landmarks, image_shape):
    if hand_landmarks is None:
        return np.zeros(21 * 3)
    
    h, w, _ = image_shape
    coords = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in hand_landmarks.landmark])

    x_min, y_min = coords[:, 0].min(), coords[:, 1].min()
    x_max, y_max = coords[:, 0].max(), coords[:, 1].max()

    width = x_max - x_min
    height = y_max - y_min

    # Evita división por cero
    if width == 0 or height == 0:
        return np.zeros(21 * 3)

    # Normaliza entre 0 y 1 dentro del bounding box
    coords[:, 0] = (coords[:, 0] - x_min) / width
    coords[:, 1] = (coords[:, 1] - y_min) / height
    coords[:, 2] = coords[:, 2] / w  # mantén z relativa al ancho

    return coords.flatten()

def extract_keypoints_normalized(results, image_shape): 
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4) 
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)  
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)  
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3) 

    
    # Normalizados los datos de las manos
    lh_normalized = normalize_hand_landmarks(results.left_hand_landmarks, image_shape) if results.left_hand_landmarks else np.zeros(21*3)
    rh_normalized = normalize_hand_landmarks(results.right_hand_landmarks, image_shape) if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh_normalized, rh_normalized])
    
    
    
