from rest_framework import permissions

class IsOwner(permissions.BasePermission):
    """
    Custom permission to only allow owners of a prediction object to view or manage it.
    """
    def has_object_permission(self, request, view, obj):
        return obj.user == request.user
