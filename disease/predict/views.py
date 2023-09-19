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

with open(str(settings.BASE_DIR) + '\predict\\trainedModel_classes\\leaf.json', 'r') as json_file:
    json_data = json.load(json_file)
# with open(str(settings.BASE_DIR) + '\predict\\trainedModel_classes\\potatoes.json', 'r') as json_file:
#     json_data = json.load(json_file)

class_names_str = json_data["CLASS_NAMES"]
CLASS_NAMES = ast.literal_eval(class_names_str)

CLASS_NAMES = {0: 'Apple___Apple_scab',
 1: 'Apple___Black_rot',
 2: 'Corn_(maize)___healthy',
 3: 'Grape___Black_rot',
 4: 'Grape___Esca_(Black_Measles)',
 5: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 7: 'Orange___Haunglongbing_(Citrus_greening)',
 8: 'Peach___Bacterial_spot',
 9: 'Peach___healthy',
 10: 'Pepper,_bell___healthy',
 11: 'Pepper,_bell___healthy',
 12: 'Apple___Cedar_apple_rust',
 13: 'Potato___Early_blight',
 14: 'Potato___Late_blight',
 16: 'Raspberry___healthy',
 17: 'Soybean___healthy',
 18: 'Squash___Powdery_mildew',
 19: 'Strawberry___Leaf_scorch',
 20: 'Strawberry___healthy',
 21: 'Tomato___Bacterial_spot',
 22: 'Tomato___Early_blight',
 23: 'Apple___healthy',
 24: 'Tomato___Late_blight',
 25: 'Tomato___Leaf_Mold',
 26: 'Tomato___Septoria_leaf_spot',
 27: 'Tomato___Spider_mites Two-spotted_spider_mite',
 28: 'Tomato___Target_Spot',
 29: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 30: 'Tomato___Tomato_mosaic_virus',
 31: 'Tomato___healthy',
 32: 'Blueberry___healthy',
 33: 'Cherry_(including_sour)___Powdery_mildew',
 34: 'Cherry_(including_sour)___healthy',
 35: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 36: 'Corn_(maize)___Common_rust_',
 37: 'Corn_(maize)___Northern_Leaf_Blight'}

 


# MODEL = tf.keras.models.load_model(str(settings.BASE_DIR) + '\predict\\trainedModels\\potatoes.h5', custom_objects={'Functional':tf.keras.models.Model})
# MODEL = tf.keras.models.load_model(str(settings.BASE_DIR) + '\predict\\trainedModels\\my_model.h5', custom_objects={'Functional':tf.keras.models.Model})
# MODEL = tf.keras.models.load_model(str(settings.BASE_DIR) + '\predict\\trainedModels\\leaf_model_256.h5', custom_objects={'Functional':tf.keras.models.Model})
# MODEL = tf.keras.models.load_model(str(settings.BASE_DIR) + '\predict\\trainedModels\\leaf_model_32_32_90.h5', custom_objects={'Functional':tf.keras.models.Model})
MODEL = tf.keras.models.load_model(str(settings.BASE_DIR) + '\predict\\trainedModels\\fine_tuned_model (1).h5', custom_objects={'Functional':tf.keras.models.Model})

urls = {
    "Apple___Apple_scab": 'https://storage.googleapis.com/kagglesdsdata/datasets/277323/658267/color/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab%203335.JPG?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20230911%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230911T193745Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=27f0291fa8d0beeb7953c68347b7fd78d7ad624da759789e13f0ff28b3e9100e8606afbd0c831e6b6d984e5084cd0e4dca80dcee65c30011c82fd0083e2f0bff1e87ffdeeb07ef48396a5b0ac6453a352e456b2377e75c867b3c1717e51c9efa05e0c4bdaecf4dcaabfc195e4819278d989694024344786213ab8b0549064387a445e48a7195082dbd3cb8927a03c770e972eb97e485ddf9f92e0a946495a9aef0a045b0ccfa1a60a5622f08ebc0b063a1fbdd486f08db5c59d2a1f9e0ef5faa0a697c02d3987bf0bc2264180210fe7ead5ef6f7f29876316c1507fbe41ab0f4fe89568e5f759a3992544ca85d6d1550866efde5fdf96cb64f199d5e6762ed69',
    "Apple___healthy": 'https://storage.googleapis.com/kagglesdsdata/datasets/277323/658267/color/Apple___healthy/0055dd26-23a7-4415-ac61-e0b44ebfaf80___RS_HL%205672.JPG?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20230911%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230911T194242Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=4858ac5aa8c1e8ce0ae89e7dfb7f4ae24da5386403743e72f77a6f6526916dccd9d6bbe307d55e75c0e2335bf06bc93bd11a7ae9ad9bbd6f3d3a2a55c60db9b19c999edb7288181a9fd61b6c895e3e12624d30e0e48dc37c27111584a2a645a7d3c5dc7183a1e3472831dd3c4f5a971092796c5574bc806ceef7733e724a4de4508ebe2d049a31cb000894dc511ee225dd08026360109cfb493b1d2daa74d0843c1b44d026c1408d3b3af056f94ebb43644fb7ee39b7b3e9dbd71a901b47caf82b3f6f53a106d8bfc7e3132dc1f6f2aeab5581cd991b0b6b11657c67937e1fbf41c8d8051f4ae5fecc8378b68a55788fe6a5bd81086fa898fa6b898c6aa9685a',
    "potato_early_blight": 'https://storage.googleapis.com/kagglesdsdata/datasets/277323/658267/color/Potato___Early_blight/00d8f10f-5038-4e0f-bb58-0b885ddc0cc5___RS_Early.B%208722.JPG?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20230914%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230914T045916Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=998d0706e48e675460488fbef19cd3c625a84c9cbfe956836b427d2c3ef2228e91ea5e4f09d0ec7ddaa0071a05673293e9558f28740c0996e3d8ea7d520f9057e252cfbfe980248576d7cc4d12e97ccb541fb9a064b799291215f51364b4ccedc55e9fc4d42d533b2300c5f3f167964efe4c59c09d1d10971b982e818e27810a088a779457a6465dc9c205697be1cc4643068a01139e59822b119726bb56381734bf3748f35bb2c46ed5bb2e339de15cac7d4b6fdb469805b84ddcc61fe865d611bc26d8b4640e4b126cc9d014c1c97087f228d2725d6a94870aa43b41b1c3cd171586759c4aa894b6bb5e8f0a0f83a931fa18f7d87b518d49bf53c4e64e5ec2',
    "apple_healthy": "https://storage.googleapis.com/kagglesdsdata/datasets/277323/658267/color/Apple___healthy/00907d8b-6ae6-4306-bfd7-d54471981a86___RS_HL%205709.JPG?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20230911%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230911T194242Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=299f862030b9ed79a670cb061a8a0eb1eaede1774b0a212028f5ae67fbd9b5f011f03e4d260f5ababa419e5dd99606ef7346c78703f2bf803da46aea811710cadc3e30e1def22fa054c81768fb2fcf76f359c21a96a7d2d09de09ee3ccc1805106bb48d31a8d90c19940c748a6bd6cd4c5f1ff089703bdf8a5a2e4386658a239b99ec553142b4c7bccfeef16c313000298897d6d1aafc1f3b357d4bab53567f972013f4485f7165a93925eb9627bb56b71a19584766c984b8d90ff60a6399546302de6b976d57eb7bc04d9fdb4bc8fd31a818d9f8fdaa2164820497420b13e16f25e07601aa145b42ee14fd0ac5aff06134841436eb144c1c301d52242955918",
    "corn_maize": 'https://storage.googleapis.com/kagglesdsdata/datasets/277323/658267/color/Corn_%28maize%29___healthy/00031d74-076e-4aef-b040-e068cd3576eb___R.S_HL%208315%20copy%202.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20230914%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230914T050242Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=641805b026eaf810b9a4560f52bceaee3015d55f6107eb1bd780851f6b4f1d8a724d51654e648816a1e74da4bece9a904b3cc70573e3db0156f687f888c485e60d8418fc18d966c55ea8a3c87b4b661a0389ae23cc1febc9337d8e384c1c50b802a69c7c51d22c17782cf3d6f1c4626fb1d14adcf306a1529f9afa93bbb966b5ddbeb9136d01de031f4d68ab42d567c52ddbe55efdfc95b0efb178132607041ab659fd7891f206bdf1849cde8721044917bd3947a72b03aff31cf1f09a60bee2377877e1776e50e1655c80e930245beeeeba3f2f1e6d512ea8ea9b41d648380bb3890556efccb6409b68ebfae51776cfa3cc34de25d31df35c6c2a0f7ee81337'
}

# Create your views here.
def home(response):
    # for name, url in urls.items():
    #     response = requests.get(url)
    #     img = Image.open(BytesIO(response.content))
    #     resized_image = img.resize((224, 224))
    #     data = asarray(resized_image)
    #     predicted_probabilities = MODEL.predict(numpy.expand_dims(data, 0))[0]
    #     predicted_class_index = numpy.argmax(predicted_probabilities)
    #     predicted_class_label = CLASS_NAMES[predicted_class_index]
    #     print("name", name, "predicted_class_label: ", predicted_class_label)
    return HttpResponse("<h1>hellos</h1>")

def number(request):
    if(request.method == 'POST'):
        number = request.POST.get('test')
        image = request.FILES['hamster']
        fss = FileSystemStorage()
        file = fss.save(image.name, image)
        file_url = fss.url(file)
        print(file_url)
        new_image_path = str(settings.BASE_DIR) + file_url

        new_image = keras_image.load_img(new_image_path, target_size=(224, 224))
        new_image_array = keras_image.img_to_array(new_image)
        new_image_array = numpy.expand_dims(new_image_array, axis=0)
        new_image_array = new_image_array / 255.0  # Normalize the image

        # Get the predicted class probabilities
        predicted_probabilities = MODEL.predict(new_image_array)[0]
        
        predicted_class_index = numpy.argmax(predicted_probabilities)
        predicted_class_label = CLASS_NAMES[predicted_class_index]
        confidence = numpy.max(predicted_probabilities[0])

        print("predicted_probabilities", predicted_probabilities)
        print("predicted_class_label", predicted_class_label)
        print("confidence", float(confidence)*100)


        return render(request, 'index.html', {"result": tools.predicting(int(number)), "number": number, "myimage": file_url,"class": predicted_class_label})



        # new_image = keras_image.load_img(new_image_path, target_size=(224, 224))
        # new_image_array = keras_image.img_to_array(new_image)
        # new_image_array = numpy.expand_dims(new_image_array, axis=0)
        # new_image_array = new_image_array / 255.0
        # predicted_probabilities = MODEL.predict(new_image_array)[0]
        # predicted_class_index = numpy.argmax(predicted_probabilities)
        # predicted_class_label = CLASS_NAMES[predicted_class_index]
        # print("predicted_class_label: ", predicted_class_label)


        # original = Image.open(str(settings.BASE_DIR) + file_url)
        # resized_image = original.resize((256, 256))
        # resized_image = original.resize((32, 32))
        # resized_image = ImageOps.grayscale(resized_image)
        # data = asarray(resized_image)
        # print("data", data)
        # predictions = MODEL.predict(numpy.expand_dims(data, 0))
        # print(predictions)
        # # prediction_class = CLASS_NAMES[numpy.argmax(predictions)]
        # prediction_class = CLASS_NAMES[numpy.argmax(predictions, axis=1)[0]]
        # confidence = numpy.max(predictions[0])

        # print(prediction_class)
        # print(confidence)

        # print(data.shape)
        # print(data)
        #  "class": prediction_class, "confidence": float(confidence)*100}
        # return render(request, 'index.html', {"result": tools.predicting(int(number)), "number": number, "myimage": file_url,"class": prediction_class, "confidence": float(confidence)*100})
        # return render(request, 'index.html', {"result": tools.predicting(int(number)), "number": number, "myimage": file_url,"class": predicted_class_label})
        return HttpResponse("<h1>hellos</h1>")

    if(request.method == 'GET'):
        return render(request, 'index.html')
    
def upload_image(request):
    if(request.method == 'POST'):
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
    else:
        form = ImageUploadForm()

    return render(request, 'upload.html', {'form': form})