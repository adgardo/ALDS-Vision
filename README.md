# ALDS-Vision
Este repositorio trata de detectar el punto de aterrizaje de un avión.
Para ello, se centra en la detección de las ruedas y determinación de rotación de las mismas.

Primero: Entrenamos el modelo de Yolo de detección de objetos para detectar ruedas.
Podemos hacer detección de objetos y segmentación. 

Segundo: generamos el dataset para entrenar las redes siamesas y la red LSTM

Tercero: hacemos doble inferencia, primero detección de ruedas, y luego analizamos la máscara de la rueda para determinar si hay rotación
