from django.urls import path
from . import views

urlpatterns = [
    path('', views.home),
    path('predict/', views.number),
    path('upload_image/', views.upload_image),
]