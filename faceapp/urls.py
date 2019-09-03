from django.urls import path,include
from .views import *

urlpatterns = [
    path('facedetect/',FaceRecognize.as_view(), name='detect'),
]
