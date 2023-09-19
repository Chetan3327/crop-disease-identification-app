from django.shortcuts import render
from django.http import HttpResponse
from . import tools
from .forms import ImageUploadForm
from django.core.files.storage import FileSystemStorage
from PIL import Image, ImageOps
from numpy import asarray
import numpy
from django.conf import settings
import tensorflow as tf
import json
import ast
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import requests
from PIL import Image
import requests
from io import BytesIO

from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.parsers import JSONParser
from rest_framework.response import Response

CLASS_NAMES = tools.get_class_names()
MODEL = tf.keras.models.load_model(str(settings.BASE_DIR) + '\predict\\trainedModels\\fine_tuned_model (1).h5', custom_objects={'Functional':tf.keras.models.Model})
print(CLASS_NAMES)
# Create your views here.

@api_view(['GET'])
def get_labels(request):
    return Response({'class_names': CLASS_NAMES})

def predict(request):
    if(request.method == 'POST'):
        image = request.FILES['uploaded_image']

        fss = FileSystemStorage()
        file = fss.save(image.name, image)
        file_url = fss.url(file)
        new_image_path = str(settings.BASE_DIR) + file_url

        new_image = keras_image.load_img(new_image_path, target_size=(224, 224))
        new_image_array = keras_image.img_to_array(new_image)
        new_image_array = numpy.expand_dims(new_image_array, axis=0)
        new_image_array = new_image_array / 255.0 

        predicted_probabilities = MODEL.predict(new_image_array)[0]
        predicted_class_index = numpy.argmax(predicted_probabilities)
        predicted_class_label = CLASS_NAMES[predicted_class_index]
        confidence = numpy.max(predicted_probabilities[0])

        print("predicted_probabilities", predicted_probabilities)
        print("predicted_class_label", predicted_class_label)
        print("confidence", float(confidence)*100)

        return render(request, 'index.html', {"image_url": file_url, "class": predicted_class_label, "confidence": confidence})

    if(request.method == 'GET'):
        return render(request, 'index.html')
    

@api_view(['POST'])
def predict_api(request):
    if(request.method == 'POST'):
        image = request.FILES['uploaded_image']

        fss = FileSystemStorage()
        file = fss.save(image.name, image)
        file_url = fss.url(file)
        new_image_path = str(settings.BASE_DIR) + file_url

        new_image = keras_image.load_img(new_image_path, target_size=(224, 224))
        new_image_array = keras_image.img_to_array(new_image)
        new_image_array = numpy.expand_dims(new_image_array, axis=0)
        new_image_array = new_image_array / 255.0 

        predicted_probabilities = MODEL.predict(new_image_array)[0]
        predicted_class_index = numpy.argmax(predicted_probabilities)
        predicted_class_label = CLASS_NAMES[str(predicted_class_index)]
        confidence = numpy.max(predicted_probabilities[0])

        print("predicted_probabilities", predicted_probabilities)
        print("predicted_class_label", predicted_class_label)
        print("confidence", float(confidence)*100)

        response_data = {
            "image_url": file_url,
            "class": predicted_class_label,
            "confidence": confidence
        }

        return Response(response_data)