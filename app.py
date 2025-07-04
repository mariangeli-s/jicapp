import cv2
import os
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask import Flask, render_template, Response

# Importar las funciones de utilidad desde el módulo detection
# Asegúrate de que 'jicapp/utils/detection.py' exista y contenga estas funciones
from utils.detection import mediapipe_detection, extract_keypoints_normalized, draw_changed_landmarks_box

# --- CONFIGURACIÓN DE LA APLICACIÓN FLASK ---
app = Flask(__name__)

# --- CONFIGURACIÓN DEL MODELO Y ACCIONES ---
MODEL_NAME = 'Modelo_Trad_Norm_Aug1.h5' 

# Construye la ruta completa al archivo del modelo de forma segura
# os.path.dirname(__file__) obtiene la ruta del directorio donde está app.py
# 'models' es el nombre de la subcarpeta donde debe estar tu modelo
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', MODEL_NAME)

# Acciones (señales) que tu modelo fue entrenado para reconocer
# ¡ASEGÚRATE DE QUE ESTAS ACCIONES COINCIDAN EXACTAMENTE CON LAS QUE USaste PARA ENTRENAR!
# El orden es crucial.
actions = np.array(['Hola', 'Gracias', 'Comprendo', 'Como estas','De nada'])

# Umbral de confianza para dar feedback positivo al usuario
# Si la probabilidad de la predicción es mayor o igual a este valor, se considera "bien hecha".
#CONFIDENCE_THRESHOLD = 0.9 # Puedes ajustar este valor según el rendimiento de tu modelo

# --- CARGAR EL MODELO ENTRENADO ---
model = None # Inicializa la variable del modelo como None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Modelo '{MODEL_NAME}' cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    print("Asegúrate de que el archivo del modelo exista en 'jicapp/models/' y la ruta sea correcta.")
    print("La aplicación continuará, pero no podrá hacer predicciones sin el modelo.")

# --- INICIALIZAR MEDIAPIPE HOLISTIC ---
# Esto se inicializa una vez al inicio de la aplicación para ser reutilizado
mp_holistic = mp.solutions.holistic

# --- FUNCIÓN GENERADORA DE FRAMES PARA EL STREAMING DE VIDEO ---
# Esta función se ejecutará cada vez que un navegador solicite el stream de video
def generate_frames():
    # Inicializa la captura de video desde la cámara (0 es la cámara por defecto)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara. Asegúrate de que no esté en uso.")
        # Podrías enviar un frame de error o un mensaje al navegador aquí si lo deseas
        return

    # Inicializa el modelo Holistic de MediaPipe para el procesamiento en tiempo real
    # Se usa un bloque 'with' para asegurar que los recursos de MediaPipe se liberen correctamente
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        sequence = [] # Lista para almacenar los keypoints de los últimos 30 frames
        sentence = [] # Lista para almacenar las predicciones de señas confirmadas (la "oración")
        predictions = [] # Lista para almacenar las últimas predicciones crudas del modelo
        threshold_prediction_consistency = 0.98 # Umbral de confianza para añadir a la 'sentence' (tu 'threshold' original)
        #frame_counter = 0 # Inicializar el contador de frames
        
        # --- NUEVA BANDERA PARA CONTROLAR EL FIN DE LA CAPTURA ---
        recognition_complete = False 
        
        # Bucle principal para leer frames de la cámara
        while True:
            ret, frame = cap.read() # Lee un frame de la cámara
            if not ret:
                print("Error: No se pudo leer el frame de la cámara. Saliendo del stream.")
                break # Sale del bucle si no se puede leer el frame

            # Voltear el frame horizontalmente para una vista de espejo (más intuitivo para el usuario)
            #frame = cv2.flip(frame, 1)
            #frame_counter += 1
            # Realizar la detección de MediaPipe utilizando la función importada
            image, results = mediapipe_detection(frame, holistic)
            #print(results)
            
            #dibujar landmarks
            draw_changed_landmarks_box(image,results)

            # Extraer keypoints del frame actual utilizando la función importada
            image_shape = image.shape
            keypoints = extract_keypoints_normalized(results,image_shape)
            sequence.append(keypoints) # Añade los keypoints al buffer de secuencia
            sequence = sequence[-30:] # Mantiene solo los últimos 30 frames (longitud de secuencia del modelo)

            # Solo hacer predicciones si tenemos suficientes frames en la secuencia y el modelo está cargado
            if len(sequence) == 30 and model is not None:
                # Realiza la predicción del modelo
                # np.expand_dims añade una dimensión de batch para que el modelo lo acepte
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                
                # Obtiene el índice de la acción predicha con mayor probabilidad
                predicted_action_index = np.argmax(res)
                # Obtiene la probabilidad de esa acción predicha
                predicted_action_prob = res[predicted_action_index]
                # Obtiene el nombre de la acción predicha
                predicted_action_name = actions[predicted_action_index]

                predictions.append(predicted_action_index) # Guarda el índice de la predicción

                # --- Lógica de visualización y feedback ---
                # Lógica para gestionar la "sentence" (historial de predicciones consistentes)
                # Si las últimas 8 predicciones son la misma seña Y la confianza es alta
                if len(predictions) >= 8 and np.unique(predictions[-8:])[0] == predicted_action_index:
                    if predicted_action_prob > threshold_prediction_consistency:
                        # Si la "sentence" está vacía o la nueva seña es diferente a la última
                        if len(sentence) == 0 or predicted_action_name != sentence[-1]:
                            sentence.append(predicted_action_name)
                            
                            # --- NUEVA LÓGICA: DETENER SI SE RECONOCE "hola" ---
                            if predicted_action_name == 'Hola': # O la seña específica que quieras
                                recognition_complete = True # Activa la bandera para detener la captura
                                print(f"¡Seña '{predicted_action_name}' reconocida! Deteniendo captura.")
                            # --------------------------------------------------
                
                # Mantener la "sentence" con un máximo de 5 señas
                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Dibujar un rectángulo en la parte superior del frame para el texto de la predicción
                #cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                # Dibujar la "oración" de señas reconocidas
                #cv2.putText(image, ' '.join(sentence), (3,30),
                           #cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

               # --- LÓGICA PARA EL FEEDBACK VISUAL BASADO EN recognition_complete ---
                feedback_text = ""
                feedback_color = (255, 255, 255) # Blanco por defecto

                if recognition_complete: # Si la seña objetivo ha sido reconocida
                    feedback_text = "BIEN HECHO! Sigue asi."
                    feedback_color = (0, 255, 0) # Verde
                else: # Si aún no se ha reconocido la seña objetivo
                    feedback_text = "Intentalo de nuevo..."
                    feedback_color = (0, 0, 255) # Rojo
                
                # Dibujar el texto de feedback en la parte inferior del frame
                cv2.putText(image, feedback_text, (150, 450), # Posición (x,y) en el frame
                           cv2.FONT_HERSHEY_SIMPLEX, 1, feedback_color, 2, cv2.LINE_AA)

            # --- DIBUJAR EL CONTADOR DE FRAMES EN PANTALLA ---
            # Posición (X, Y) para el contador de frames
            # Puedes ajustar estas coordenadas para que no se superpongan con otros elementos
            # Por ejemplo, (50, 100) como sugeriste, o (image.shape[1] - 150, 60) para la esquina superior derecha
            #counter_text_pos = (3, 50) # (X, Y)
            #cv2.putText(image, f'Frame #{frame_counter}', counter_text_pos,
                        #cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            
            # --- LÓGICA PARA DETENER LA CAPTURA Y MOSTRAR MENSAJE FINAL ---
            if recognition_complete:
                # Opcional: Puedes dibujar un mensaje final en el frame antes de salir
                final_message = "RECONOCIMIENTO DE HOLA COMPLETADO"
                text_size = cv2.getTextSize(final_message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (image.shape[1] - text_size[0]) // 2
                text_y = (image.shape[0] + text_size[1]) // 2
                cv2.putText(image, final_message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) # Verdec
            
                # Codificar y enviar el último frame con el mensaje final
                ret, buffer = cv2.imencode('.jpg', image)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Romper el bucle principal
                break 
            # -------------------------------------------------------------

            # Codificar el frame procesado como una imagen JPEG para el streaming
            ret, buffer = cv2.imencode('.jpg', image)
            frame_bytes = buffer.tobytes()

            # Enviar el frame codificado al cliente como parte de un stream multipart
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release() # Libera los recursos de la cámara cuando el generador termina
    print("Captura de cámara finalizada debido a reconocimiento de seña.")
    
# FUNCION PARA COMPRENDO...
def generate_frames_comprendo():
    # Inicializa la captura de video desde la cámara (0 es la cámara por defecto)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara. Asegúrate de que no esté en uso.")
        # Podrías enviar un frame de error o un mensaje al navegador aquí si lo deseas
        return

    # Inicializa el modelo Holistic de MediaPipe para el procesamiento en tiempo real
    # Se usa un bloque 'with' para asegurar que los recursos de MediaPipe se liberen correctamente
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        sequence = [] # Lista para almacenar los keypoints de los últimos 30 frames
        sentence = [] # Lista para almacenar las predicciones de señas confirmadas (la "oración")
        predictions = [] # Lista para almacenar las últimas predicciones crudas del modelo
        threshold_prediction_consistency = 0.8 # Umbral de confianza para añadir a la 'sentence' (tu 'threshold' original)
        #frame_counter = 0 # Inicializar el contador de frames
        
        # --- NUEVA BANDERA PARA CONTROLAR EL FIN DE LA CAPTURA ---
        recognition_complete = False 
        
        # Bucle principal para leer frames de la cámara
        while True:
            ret, frame = cap.read() # Lee un frame de la cámara
            if not ret:
                print("Error: No se pudo leer el frame de la cámara. Saliendo del stream.")
                break # Sale del bucle si no se puede leer el frame

            # Realizar la detección de MediaPipe utilizando la función importada
            image, results = mediapipe_detection(frame, holistic)
            #print(results)
            
            #dibujar landmarks
            draw_changed_landmarks_box(image,results)

            # Extraer keypoints del frame actual utilizando la función importada
            image_shape = image.shape
            keypoints = extract_keypoints_normalized(results,image_shape)
            sequence.append(keypoints) # Añade los keypoints al buffer de secuencia
            sequence = sequence[-30:] # Mantiene solo los últimos 30 frames (longitud de secuencia del modelo)

            # Solo hacer predicciones si tenemos suficientes frames en la secuencia y el modelo está cargado
            if len(sequence) == 30 and model is not None:
                # Realiza la predicción del modelo
                # np.expand_dims añade una dimensión de batch para que el modelo lo acepte
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                
                # Obtiene el índice de la acción predicha con mayor probabilidad
                predicted_action_index = np.argmax(res)
                # Obtiene la probabilidad de esa acción predicha
                predicted_action_prob = res[predicted_action_index]
                # Obtiene el nombre de la acción predicha
                predicted_action_name = actions[predicted_action_index]

                predictions.append(predicted_action_index) # Guarda el índice de la predicción

                # --- Lógica de visualización y feedback ---
                # Lógica para gestionar la "sentence" (historial de predicciones consistentes)
                # Si las últimas 8 predicciones son la misma seña Y la confianza es alta
                if len(predictions) >= 4 and np.unique(predictions[-4:])[0] == predicted_action_index:
                    if predicted_action_prob > threshold_prediction_consistency:
                        # Si la "sentence" está vacía o la nueva seña es diferente a la última
                        if len(sentence) == 0 or predicted_action_name != sentence[-1]:
                            sentence.append(predicted_action_name)
                            
                            # --- NUEVA LÓGICA: DETENER SI SE RECONOCE "hola" ---
                            if predicted_action_name == 'Comprendo': # O la seña específica que quieras
                                recognition_complete = True # Activa la bandera para detener la captura
                                print(f"¡Seña '{predicted_action_name}' reconocida! Deteniendo captura.")
                            # --------------------------------------------------
                
                # Mantener la "sentence" con un máximo de 5 señas
                if len(sentence) > 5:
                    sentence = sentence[-5:]

               # --- LÓGICA PARA EL FEEDBACK VISUAL BASADO EN recognition_complete ---
                feedback_text = ""
                feedback_color = (255, 255, 255) # Blanco por defecto

                if recognition_complete: # Si la seña objetivo ha sido reconocida
                    feedback_text = "BIEN HECHO! Sigue asi."
                    feedback_color = (0, 255, 0) # Verde
                else: # Si aún no se ha reconocido la seña objetivo
                    feedback_text = "Intentalo de nuevo..."
                    feedback_color = (0, 0, 255) # Rojo
                
                # Dibujar el texto de feedback en la parte inferior del frame
                cv2.putText(image, feedback_text, (150, 450), # Posición (x,y) en el frame
                           cv2.FONT_HERSHEY_SIMPLEX, 1, feedback_color, 2, cv2.LINE_AA)
            
            # --- LÓGICA PARA DETENER LA CAPTURA Y MOSTRAR MENSAJE FINAL ---
            if recognition_complete:
                # Opcional: Puedes dibujar un mensaje final en el frame antes de salir
                final_message = "RECONOCIMIENTO DE COMPRENDO COMPLETADO"
                text_size = cv2.getTextSize(final_message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (image.shape[1] - text_size[0]) // 2
                text_y = (image.shape[0] + text_size[1]) // 2
                cv2.putText(image, final_message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) # Verdec
            
                # Codificar y enviar el último frame con el mensaje final
                ret, buffer = cv2.imencode('.jpg', image)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Romper el bucle principal
                break 
            # -------------------------------------------------------------

            # Codificar el frame procesado como una imagen JPEG para el streaming
            ret, buffer = cv2.imencode('.jpg', image)
            frame_bytes = buffer.tobytes()

            # Enviar el frame codificado al cliente como parte de un stream multipart
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release() # Libera los recursos de la cámara cuando el generador termina
    print("Captura de cámara finalizada debido a reconocimiento de seña.")

# --- RUTAS DE FLASK ---
# Define la ruta principal de la aplicación web (ej. http://127.0.0.1:5000/)
@app.route('/')
def index():
    # Renderiza la plantilla HTML principal (index.html) que el usuario verá
    return render_template('lecciones.html')

# Define la ruta para el stream de video de la cámara
@app.route('/captando')
def video_feed():
    #Esta ruta sirve el stream de video generado por la función generate_frames()
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/comprendo')
def comprendo():
    return render_template('comprendo.html')

@app.route('/captacomprendo')
def video_feed_comprendo():
    #Esta ruta sirve el stream de video generado por la función generate_frames()
    return Response(generate_frames_comprendo(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- EJECUTAR LA APLICACIÓN ---
# Este bloque se ejecuta solo si el script se inicia directamente (no si se importa como módulo)
if __name__ == '__main__':
    # Inicia el servidor de desarrollo de Flask
    # debug=True es útil durante el desarrollo para recargar automáticamente y mostrar errores
    # ¡No usar debug=True en producción!
    app.run(debug=True)
