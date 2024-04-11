from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from ultralytics import YOLO
# Importar las librerías necesarias para la red siamesa
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Lambda, Dense, Dropout
import csv
# Definir la función de distancia euclidiana
def euclidean_distance(vectors):
    # Desempaquetar los vectores
    x, y = vectors
    # Calcular la suma de las diferencias al cuadrado
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    # Aplicar la raíz cuadrada y devolver el resultado
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

# Definir la entrada de la red siamesa
input_shape = (32, 32, 3) # Usar el mismo tamaño que las imágenes
left_input = Input(input_shape) # Entrada izquierda
right_input = Input(input_shape) # Entrada derecha

# Definir la subred base
base_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(6, (5, 5), activation="tanh", input_shape=input_shape), # Primera capa convolucional
    tf.keras.layers.AveragePooling2D((2, 2)), # Primera capa de agrupación promedio
    tf.keras.layers.Dropout(0.2), # Dropout
    tf.keras.layers.Conv2D(16, (5, 5), activation="tanh"), # Segunda capa convolucional
    tf.keras.layers.AveragePooling2D((2, 2)), # Segunda capa de agrupación promedio
    tf.keras.layers.Dropout(0.2), # Dropout
    tf.keras.layers.Flatten(), # Capa para aplanar las características
    tf.keras.layers.Dense(120, activation="tanh"), # Primera capa densa
    tf.keras.layers.Dense(84, activation="tanh") # Segunda capa densa
]) # Crear LeNet-5 como subred base
# Extraer las características de cada entrada usando la subred base
left_features = base_model(left_input)
right_features = base_model(right_input)

# Calcular la distancia entre las características usando una capa Lambda
distance = Lambda(euclidean_distance)([left_features, right_features])

# Aplicar una capa densa con activación softmax para obtener la probabilidad de cada clase
output = Dense(3, activation="softmax")(distance)

# Crear el modelo de la red siamesa
model = Model(inputs=[left_input, right_input], outputs=output)

# Cargar los pesos al modelo
model.load_weights("siam_weights.h5")

# Definir la función para la comparación de imágenes con la red siamesa
def comparar_imagenes(img1, img2):

    # Redimensionar las imágenes a 32 x 32
    img1_resized = cv2.resize(img1, (32, 32))
    img2_resized = cv2.resize(img2, (32, 32))

    # Añadir una dimensión extra al principio de la entrada
    img1_expanded = np.expand_dims(img1_resized, axis=0)
    img2_expanded = np.expand_dims(img2_resized, axis=0)

    # Añadir una dimensión extra para los canales de color
    img1_expanded = np.expand_dims(img1_expanded, axis=-1)
    img2_expanded = np.expand_dims(img2_expanded, axis=-1)

    # Hacer la predicción con la red siamesa
    pred = model.predict([img1_expanded, img2_expanded])
    # Obtener la clase predicha

    # print(f'Siamese result for track {track_id}: {pred}')
    # print(framecount)
    #
    # plt.figure(figsize=(20,20))
    # # Muestra la imagen annotated_frame
    # plt.subplot(1, 2, 1)  # 1 fila, 2 columnas, índice de imagen 1
    # plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    # plt.title('img1')
    #
    # # Muestra la imagen frame
    # plt.subplot(1, 2, 2)  # 1 fila, 2 columnas, índice de imagen 2
    # plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    # plt.title('img2')
    # plt.show()
    return pred

# Carga el modelo YOLOv8
modelo = YOLO('best_ins_seg.pt')

nombre_video= 'Prueba 7.MOV'
# Abre el archivo de video
cap = cv2.VideoCapture(nombre_video)


# Obtiene las dimensiones del video
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Configura el archivo de video de salida
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(nombre_video+'output.avi', fourcc, fps, (1440, 300))

# Almacena el historial de seguimiento y las imágenes anteriores
track_history = defaultdict(lambda: [])
previous_frames = defaultdict(lambda: None)
framecount= 0
colors = [
    (255, 0, 0),(0, 0, 255),(0, 128, 0),(255, 255, 0),(128, 0, 128),(255, 165, 0),(0, 0, 0),(255, 255, 255),(255, 192, 203),(165, 42, 42),(128, 128, 128),(0, 255, 255),
    (255, 0, 255),(0, 255, 0),(128, 0, 0),(255, 20, 147),(255, 140, 0),(75, 0, 130),(127, 255, 0),(210, 105, 30),(100, 149, 237),(189, 183, 107),(255, 105, 180),
    (173, 216, 230),(32, 178, 170),(147, 112, 219),(60, 179, 113),(123, 104, 238),(0, 206, 209),(138, 43, 226)
]
# Bucle a través de los fotogramas del video
inicio = time.time()
while cap.isOpened():
    # Lee un fotograma del video
    success, frame = cap.read()
    framecount = framecount +1
    if success:
        # Preprocesamiento
        frame = frame[int(0.48*1080):int(0.75*1080), :]
        frame = cv2.resize(frame, (1440, 300))

        # Ejecuta el seguimiento YOLOv8 en el fotograma, persistiendo las pistas entre fotogramas
        results = modelo.track(frame, max_det = 2, show = False, persist=True, tracker="bytetrack2.yaml")

        cv2.putText(frame, str(framecount), (50, 25), cv2.FONT_HERSHEY_SIMPLEX,1,(200, 200, 200))
        try:
            # Obtiene los cuadros y los ID de seguimiento
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            masks = results[0].masks.cpu().data.numpy().transpose(1, 2, 0)

            # Traza las pistas
            for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                x, y, w, h = map(int, box)
                if x-w<0:
                    pass
                else:
                    track = track_history[track_id]
                    mask_single = masks[:, :, i]  # Get the i-th mask
                    mask_single = cv2.resize(mask_single,
                                             (frame.shape[1], frame.shape[0]))  # Resize the mask to the original frame size
                    mask_single = (mask_single * 255).astype(np.uint8)  # Convert the mask to uint8
                    masked_frame = cv2.bitwise_and(frame, frame, mask=mask_single)
                    current_frame = masked_frame
                    current_frame = current_frame[y - int(w / 2):y + int(h / 2), x - int(h / 2):x + int(h / 2)]

                    # Si hay un frame anterior para este track_id, haz la inferencia de la red siamesa
                    if previous_frames[track_id] is not None:
                        previous_frame = previous_frames[track_id]
                        predict = comparar_imagenes(previous_frame, current_frame)
                        prediction = np.argmax(predict)
                        conf = results[0].boxes[i].conf.item()
                        conf_str = str(round(conf, 2))
                        text = str(predict) + " " + conf_str
                        if track_id == min(track_ids):
                            pos = 0
                        else:
                            pos = 1
                        cv2.putText(frame, text, (100, 25 + (250*pos)), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[track_id])

                        #print(f'Siamese result for track {track_id}: {prediction}')
                    else:
                        prediction = 0  # asumimos que la rueda no está rodando si no hay frame anterior
                    if prediction == 1:
                        color = colors[track_id]
                    else:
                        # Reduce la intensidad a la mitad
                        color = tuple([int(c * 0.5) for c in colors[track_id]])
                    cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                                  color, 3)

                    # Guarda la predicción junto con las coordenadas en la traza
                    track.append((float(x), float(y), float(h), float(w), prediction, framecount, track_id))

                    # Dibuja las líneas de seguimiento
                    for j in range(1, len(track)):
                        x, y, h, w, prediction, _ , _= track[j]
                        x_prev, y_prev, h_prev, w_prev, prediction_prev, _ , _= track[j - 1]
                        if prediction_prev == 1:
                            color = colors[track_id]
                        else:
                            color = tuple([int(c * 0.5) for c in colors[track_id]])
                        pt1 = (int(x_prev), int(y_prev+h_prev/2))
                        pt2 = (int(x), int(y+h/2))
                        cv2.line(frame, pt1, pt2, color, thickness=3)

                    # Almacena el frame actual como el frame anterior para este track_id
                    previous_frames[track_id] = current_frame

            # Muestra el fotograma anotado
            out.write(frame)
            filename = "frame" + str(framecount) + ".jpg"
            #cv2.imwrite(filename, frame)
        except AttributeError:
            pass

        # Interrumpe el bucle si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Interrumpe el bucle si se alcanza el final del video
        break
fin = time.time()
print(fin-inicio)

for track_id, track in track_history.items():
    # Abre un archivo en modo escritura ('w')
    with open(nombre_video+str(track_id)+'SIAM_track.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        # Escribe la cabecera del CSV
        writer.writerow(["x", "y", "h", "w", "prediction", "framecount", "track_id"])
        # Escribe cada tupla en una nueva línea del CSV
        for row in track:
            # Reemplaza el punto decimal por una coma
            writer.writerow([str(item).replace('.', ',') for item in row])

# Libera el objeto de captura de video y cierra la ventana de visualización
out.release()
cap.release()
cv2.destroyAllWindows()