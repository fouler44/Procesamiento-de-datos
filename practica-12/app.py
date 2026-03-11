import math
import logging
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from urllib import parse
from http.server import HTTPServer, BaseHTTPRequestHandler


# 1) Ajuste de mensajes de TensorFlow

tf_logger = tf.get_logger()
tf_logger.setLevel(logging.ERROR)


# 2) Carga del conjunto de datos MNIST

print("Preparando dataset MNIST...")

datos, info = tfds.load(
    'mnist',
    as_supervised=True,
    with_info=True
)

datos_entrenamiento = datos['train']
datos_prueba = datos['test']

etiquetas_texto = [
    'Cero', 'Uno', 'Dos', 'Tres', 'Cuatro',
    'Cinco', 'Seis', 'Siete', 'Ocho', 'Nueve'
]

total_entrenamiento = info.splits['train'].num_examples
total_pruebas = info.splits['test'].num_examples


# 3) Escalado de imagenes

def escalar_imagen(imagen, etiqueta):
    imagen = tf.cast(imagen, tf.float32) / 255.0
    return imagen, etiqueta

datos_entrenamiento = datos_entrenamiento.map(escalar_imagen).cache()
datos_prueba = datos_prueba.map(escalar_imagen).cache()


# 4) Construccion del modelo

red_neuronal = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

red_neuronal.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# 5) Organizacion en lotes

TAM_LOTE = 32

datos_entrenamiento = (
    datos_entrenamiento
    .repeat()
    .shuffle(total_entrenamiento)
    .batch(TAM_LOTE)
)

datos_prueba = datos_prueba.batch(TAM_LOTE)


# 6) Entrenamiento del modelo

print("Iniciando entrenamiento...")
red_neuronal.fit(
    datos_entrenamiento,
    epochs=5,
    steps_per_epoch=math.ceil(total_entrenamiento / TAM_LOTE)
)


# 7) Medicion del rendimiento

perdida, precision = red_neuronal.evaluate(
    datos_prueba,
    steps=math.ceil(total_pruebas / TAM_LOTE)
)

print("Precision obtenida en prueba:", precision)


# 8) Servidor para recibir pixeles

class ManejadorNumeros(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        print("Solicitud POST detectada")

        try:
            # Leer el contenido enviado por el cliente
            longitud = int(self.headers['Content-Length'])
            contenido = self.rfile.read(longitud).decode()

            # Quitar el nombre del parametro y decodificar caracteres URL
            contenido = contenido.replace('pixeles=', '')
            contenido = parse.unquote(contenido)

            # Convertir la cadena en arreglo numerico
            pixeles = np.fromstring(contenido, dtype=np.float32, sep=",")

            if pixeles.size != 784:
                raise ValueError(f"Se recibieron {pixeles.size} valores, pero se esperaban 784")

            # Dar formato correcto para el modelo
            imagen = pixeles.reshape(28, 28)
            imagen = np.array(imagen, dtype=np.float32)
            imagen = imagen.reshape(1, 28, 28, 1)

            # Clasificacion
            resultado = red_neuronal.predict(imagen, batch_size=1, verbose=0)
            numero_predicho = int(np.argmax(resultado))
            nombre_numero = etiquetas_texto[numero_predicho]

            print("Resultado:", numero_predicho, "-", nombre_numero)

            texto_respuesta = f"{numero_predicho} ({nombre_numero})"

            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(texto_respuesta.encode("utf-8"))

        except Exception as error:
            print("Ocurrio un error:", str(error))
            self.send_response(500)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(f"Error: {str(error)}".encode("utf-8"))


# 9) Ejecucion del servidor local

print("Servidor activo en http://localhost:8000 ...")
servidor_local = HTTPServer(('localhost', 8000), ManejadorNumeros)
servidor_local.serve_forever()