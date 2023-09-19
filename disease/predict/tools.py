import ast
import json
from django.conf import settings

def get_class_names():
    with open(str(settings.BASE_DIR) + '\predict\\trainedModel_classes\\leaf_labels.json', 'r') as json_file:
        CLASS_NAMES = json.load(json_file)
    return CLASS_NAMES