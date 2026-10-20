from django.urls import path
from core import views

urlpatterns = [
    path('', views.home, name='home'),
    path('search/', views.search, name='search'),
    path('post/<int:post_id>/', views.post_detail, name='post_detail'),
    path('recommendations/', views.recommendations, name='recommendations'),
]