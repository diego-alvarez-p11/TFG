# Diseño e implementación de un prototipo de sistema de ayuda a la conducción en vehículos basado en detección de objetos empleando visión artificial


El objetivo del proyecto es realizar el diseño y la implementación de un prototipo de sistema de ayuda a la conducción en vehículos basado en detección de objetos empleando visión artificial con Deep Learning orientado a la monitorización del entorno en un vehículo autónomo. El sistema será capaz de analizar el vídeo captado por una cámara frontal mientras se mueve el vehículo para reconocer los elementos más importantes de la vía que puedan resultar relevantes para la conducción, como otros vehículos, peatones, semáforos, señales y otras circunstancias de alerta.

![image](https://user-images.githubusercontent.com/54302649/133695967-b8c64be0-2580-4afd-9062-b56683dbbc95.png)


El algoritmo utilizado para la construcción de este sistema ha sido You Only Look Once (YOLOv5), desarrollado por Ultralytics. Este algoritmo proporciona un gran rendimiento en la detección de objetos en tiempo real, haciendo uso de capas convolucionales. Las ventajas que lo diferencian del resto de modelos son su rapidez en el procesado y su alta precisión en la detección final.

![Backbone](https://user-images.githubusercontent.com/54302649/133696566-7be1ec55-f9f9-4d8a-a436-9b88c22404df.png)


Para la implementación del prototipo se ha utilizado un triple detector sobre cada fotograma. Cada uno de dichos detectores se encarga del reconocimiento de un grupo de objetos: el primero hace uso del conjunto de datos COCO y sus correspondientes pesos pre-entrenados para la detección de elementos dinámicos como vehículos y personas; el
segundo se ha entrenado con el conjunto de datos de Bosch con el fin de diferenciar la luz iluminada en semáforos y su flecha de dirección y, finalmente, el tercer detector se aplica para el reconocimiento de señales de tráfico. De cara a la obtención de las imágenes necesarias para realizar el entrenamiento de este último detector, se ha combinado el conjunto de datos de German Traffic Sign Detection Benchmark (GTSDB) con imágenes etiquetadas manualmente mediante la herramienta LabelImg.

<img src="https://user-images.githubusercontent.com/54302649/133697928-c93a8b7b-a412-45c9-89e8-edcada5a4049.png" width="800" height="250">


Para concluir, una vez comprobado el correcto funcionamiento, se ha incorporado un módulo encargado del sistema de ayuda a la conducción. Este muestra por pantalla la
velocidad máxima de la vía con la ayuda de las señales de velocidad detectadas. En caso de encontrar una señal de Stop, una señal de paso de peatones junto con la detección de un viandante, un semáforo en rojo o en caso de detectarse la presencia de obstáculos en la trayectoria, este valor se reduce. Además, se muestran los iconos de las señales detectadas en el lado correspondiente de la vía, así como los semáforos en la parte superior. Adicionalmente, se ha agregado un sistema de alerta por proximidad.
