from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from django.contrib.auth import authenticate, login as django_login, logout as django_logout
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import RegisterSerializer, UserSerializer

class RegisterView(generics.CreateAPIView):
    serializer_class = RegisterSerializer
    permission_classes = [AllowAny]

    def get(self, request, *args, **kwargs):
        """Render the registration HTML page for GET requests."""
        return render(request, 'accounts/register.html')

class LoginView(APIView):
    permission_classes = [AllowAny]

    def get(self, request, *args, **kwargs):
        """Render the login HTML page for GET requests."""
        return render(request, 'accounts/login.html')

    def post(self, request, *args, **kwargs):
        """Authenticate user and return JWT tokens on POST requests."""
        username = request.data.get('username')
        password = request.data.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            django_login(request, user)
            refresh = RefreshToken.for_user(user)
            return Response({
                'refresh': str(refresh),
                'access': str(refresh.access_token),
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'bio': user.bio,
                    'profile_picture': user.profile_picture.url if user.profile_picture else None
                }
            }, status=status.HTTP_200_OK)
        else:
            return Response(
                {'error': 'Invalid username or password'},
                status=status.HTTP_401_UNAUTHORIZED
            )

class LogoutView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        """Blacklist the refresh token to logout the user."""
        try:
            django_logout(request)
            refresh_token = request.data.get('refresh')
            if not refresh_token:
                return Response(
                    {'error': 'Refresh token is required'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            token = RefreshToken(refresh_token)
            token.blacklist()
            return Response(
                {'message': 'Successfully logged out'},
                status=status.HTTP_205_RESET_CONTENT
            )
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

class UserProfileView(generics.RetrieveUpdateAPIView):
    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated]

    def get_object(self):
        """Return the authenticated user's profile."""
        return self.request.user

class ChangePasswordView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        old_password = request.data.get('old_password')
        new_password = request.data.get('new_password')
        confirm_password = request.data.get('confirm_password')

        if not old_password or not new_password or not confirm_password:
            return Response({'error': 'All password fields are required.'}, status=status.HTTP_400_BAD_REQUEST)

        if not user.check_password(old_password):
            return Response({'error': 'Incorrect current password.'}, status=status.HTTP_400_BAD_REQUEST)

        if new_password != confirm_password:
            return Response({'error': 'New passwords do not match.'}, status=status.HTTP_400_BAD_REQUEST)

        from django.contrib.auth.password_validation import validate_password
        from django.core.exceptions import ValidationError
        try:
            validate_password(new_password, user)
        except ValidationError as e:
            return Response({'error': e.messages[0]}, status=status.HTTP_400_BAD_REQUEST)

        user.set_password(new_password)
        user.save()
        
        from django.contrib.auth import update_session_auth_hash
        update_session_auth_hash(request, user)

        return Response({'message': 'Password updated successfully.'}, status=status.HTTP_200_OK)

@login_required(login_url='/accounts/login/')
def profile_view(request):
    """
    Renders the clinician's profile page.
    """
    return render(request, 'accounts/profile.html')


