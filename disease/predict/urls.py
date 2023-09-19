from django.urls import path
from . import views
from . import tools
from django.http import HttpResponse

urlpatterns = [
    path('get_labels/', views.get_labels),
    path('predict_api/', views.predict_api),
    path('predict/', views.predict),
]