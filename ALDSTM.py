from collections import defaultdict
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from keras.models import load_model
import time
import csv

# Definir
# Cargar los pesos al modelo
modelLSTM= load_model('LSTM.h5')

# Carga el modelo YOLOv8
modelo = YOLO('best_ins_seg.pt')

nombre_video= 'Prueba 15.MOV'
# Abre el archivo de video
cap = cv2.VideoCapture(nombre_video)

# Obtiene las dimensiones del video
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Configura el archivo de video de salida
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(nombre_video+'.avi', fourcc, fps, (1440, 300))
# Luego, creamos un diccionario para almacenar los últimos 5 frames de cada track_id
buffer = {}

# Almacena el historial de seguimiento y las imágenes anteriores
track_history = defaultdict(lambda: [])
previous_frames = defaultdict(lambda: None)
framecount= 0
colors = [
    (255, 255, 0),(128, 0, 128),(255, 165, 0),(173, 216, 230),(147, 112, 219),(0, 206, 209),(138, 43, 226),(255, 0, 0),(0, 0, 255),(0, 128, 0),(0, 0, 0),(255, 255, 255),(255, 192, 203),(165, 42, 42),(128, 128, 128),(0, 255, 255),
    (255, 0, 255),(0, 255, 0),(128, 0, 0),(255, 20, 147),(255, 140, 0),(75, 0, 130),(127, 255, 0),(210, 105, 30),(100, 149, 237),(189, 183, 107),(255, 105, 180),
    (32, 178, 170),(60, 179, 113),(123, 104, 238)
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

                        # Leer la imagen y convertirla a escala de grises
                        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                        # Redimensionar la imagen a 32x32 píxeles
                        current_frame = cv2.resize(current_frame, (32, 32))
                        # Normalizar la imagen
                        current_frame = current_frame / 255.0
                        # Aplanar la imagen
                        current_frame = current_frame.flatten()
                        # Añadir el frame al buffer del track_id correspondiente
                        if track_id not in buffer:
                            buffer[track_id] = []
                        buffer[track_id].append(current_frame)
                        # Si el buffer para este track_id tiene más de 5 frames, eliminamos el más antiguo
                        if len(buffer[track_id]) > 5:
                            buffer[track_id].pop(0)

                        # Si el buffer para este track_id tiene exactamente 5 frames, hacemos la inferencia
                        if len(buffer[track_id]) == 5:
                            sequence = np.array([buffer[track_id]])
                            predict = modelLSTM.predict(sequence)
                            prediction = predict[0][0]
                            prediction = round(prediction, 2)
                            print(f"Prediction for track {track_id}: {prediction}")
                            conf = results[0].boxes[i].conf.item()
                            conf_str = str(round(conf, 2))
                            text = "     Pred: "+ str(prediction) + " Conf: " + conf_str
                            if track_id == min(track_ids):
                                pos = 0
                            else:
                                pos = 1
                            cv2.putText(frame, text, (100, 25 + (250*pos)), cv2.FONT_HERSHEY_SIMPLEX, 1   , colors[track_id])

                        else:
                            prediction = 0

                    else:
                        prediction = 0  # asumimos que la rueda no está rodando si no hay buffer de 5
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
    with open(nombre_video+str(track_id)+'LSTM_track.csv', 'w', newline='') as file:
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
