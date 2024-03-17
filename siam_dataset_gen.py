import os
import csv
from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

# Carga el modelo YOLOv8
model = YOLO('best.pt')

videoname= '../Prueba 7.MOV'
# Abre el archivo de video
cap = cv2.VideoCapture(videoname)

# Crea una carpeta para guardar las imágenes de las ruedas
os.makedirs('ruedas', exist_ok=True)

# Inicializa las variables para seguir el estado de las ruedas
rueda_giro = defaultdict(int)

# Inicializa un diccionario para guardar las imágenes de las ruedas
ruedas = defaultdict(list)
size = int(32/2)
factor_ampliacion = 8

# Loop through the video frames
frame_count = 0
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Preprocesamiento
        frame = frame[int(0.5*1080):int(0.75*1080), :]
        frame = cv2.resize(frame, (1440, 256))

        # Run YOLOv8 tracking on the frame
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        # Verifica si se detectaron objetos
        try:
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Loop through the detected boxes
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = map(int, box)
                rueda_img = frame[y-size:y+size, x-size:x+size]

                # Check if the wheel started spinning
                if rueda_giro[track_id] == 0:
                    # Show the current wheel image and the previous one
                    if rueda_img.size == 0:
                        print(f"La imagen de la rueda está vacía para el track_id {track_id} en el frame {frame_count}.")
                    else:
                        rueda_img_grande = cv2.resize(rueda_img, (rueda_img.shape[1]*factor_ampliacion, rueda_img.shape[0]*factor_ampliacion))
                        cv2.imshow('Rueda actual', rueda_img_grande)

                    if ruedas[track_id]:
                        if len(ruedas[track_id]) > 0 and ruedas[track_id][-1][2].size != 0:
                            rueda_anterior_grande = cv2.resize(ruedas[track_id][-1][2], (ruedas[track_id][-1][2].shape[1]*factor_ampliacion, ruedas[track_id][-1][2].shape[0]*factor_ampliacion))
                            cv2.imshow('Rueda anterior', rueda_anterior_grande)
                        else:
                            print(f"No hay imagen guardada para la rueda con track_id {track_id} en el frame {frame_count}.")
                    cv2.waitKey(1)
                    giro = input(f'Rueda {track_id} giró? ')
                    if giro == '1':
                        rueda_giro[track_id] = 1

                # Save the wheel image
                img_filename = f'{videoname}_rueda_{track_id}_frame_{frame_count}_{rueda_giro[track_id]}.jpg'
                cv2.imwrite(os.path.join('ruedas', img_filename), rueda_img)

                # Add the wheel image to the dictionary
                ruedas[track_id].append((frame_count, rueda_giro[track_id], rueda_img))

        except AttributeError:
            pass

        # Increment the frame count
        frame_count += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
