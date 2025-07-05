# api/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import tensorflow as tf
import numpy as np
import time
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
import io


model_1 = load_model('../Modelo_1/modelo_hojas_multiclasev3.h5')
model_2 = load_model('../Modelo_2/modelo_plagas_binario.h5')
model_3 = load_model('../Modelo_3/modelo_plagas_multiclase.h5')


clases_hojas = ['hojas', 'otros', 'animales'] 
clases_plaga = ['no_plaga', 'plaga'] 
clases_plaga_detalle = ['liriomyza', 'rust', 'bacterial_spot', 'otros']


def preprocess_image(img_path, target_size=(224, 224)):
    img = keras_image.load_img(img_path, target_size=target_size)
    img_array = keras_image.img_to_array(img) / 255.0  
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return img_array_expanded

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:

            start_time = time.time()

            img_file = request.FILES['image']
            img_path = 'tmp_image.jpg'  
            with open(img_path, 'wb') as f:
                for chunk in img_file.chunks():
                    f.write(chunk)


            img_array = preprocess_image(img_path)
            prediction_hoja = model_1.predict(img_array)
            class_hoja = np.argmax(prediction_hoja, axis=1)[0]
            label_hoja = clases_hojas[class_hoja]
            hoja_precision = float(prediction_hoja[0][class_hoja])

            if label_hoja == 'otros' or label_hoja == 'animales':
                return JsonResponse({
                    'status': label_hoja,
                    'message': 'No se detectó hoja',
                    'precision': round(hoja_precision, 2),
                    'tiempo': round(time.time() - start_time, 2),
                })

            prediction_plaga = model_2.predict(img_array)
            is_plaga = np.argmax(prediction_plaga, axis=1)[0]
            plaga_probability = prediction_plaga[0][0]
            label_plaga = 'plaga' if plaga_probability > 0.5 else 'no_plaga'
            plaga_precision = float(prediction_plaga[0][is_plaga])

            if label_plaga == 'no_plaga':
                return JsonResponse({
                    'status': label_plaga,
                    'message': 'No es plaga',
                    'precision': round(plaga_precision, 2),
                    'tiempo': round(time.time() - start_time, 2)
                })


            prediction_plaga_detalle = model_3.predict(img_array)
            plaga_detalle_class = np.argmax(prediction_plaga_detalle, axis=1)[0]
            label_plaga_detalle = clases_plaga_detalle[plaga_detalle_class]
            plaga_detalle_precision = float(prediction_plaga_detalle[0][plaga_detalle_class])


            elapsed_time = time.time() - start_time


            response = {
                'status': 'success',
                'hoja_predicha': label_hoja,
                'hoja_precision': round(hoja_precision, 2),
                'plaga_predicha': label_plaga,
                'plaga_precision': round(plaga_precision, 2),
                'plaga_detalle_predicha': label_plaga_detalle,
                'plaga_detalle_precision': round(plaga_detalle_precision, 2),
                'tiempo': round(elapsed_time, 2)
            }
            return JsonResponse(response)

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})

    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})
