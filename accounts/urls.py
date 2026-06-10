from django.urls import path
from .views import RegisterView, LoginView, LogoutView, profile_view, UserProfileView, ChangePasswordView

app_name = 'accounts'

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('profile/', profile_view, name='profile'),
    path('profile/api/', UserProfileView.as_view(), name='profile-api'),
    path('profile/password/', ChangePasswordView.as_view(), name='change-password'),
]

