from types import SimpleNamespace

class IterableSimpleNamespace(SimpleNamespace):
    """IterableSimpleNamespace is an extension class of SimpleNamespace that adds iterable functionality and
        enables usage with dict() and for loops.
    """

    def __iter__(self):
        """Return an iterator of key-value pairs from the namespace's attributes."""
        return iter(vars(self).items())

    def __str__(self):
        """Return a human-readable string representation of the object."""
        return "\n".join(f"{k}={v}" for k, v in vars(self).items())

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(
            f"'{name}' object has no attribute '{attr}'. To access, use dict() or iterate over the object."
        )

    def get(self, key, default=None):
        """Return the value of the specified key if it exists; otherwise, return the default value."""
        return getattr(self, key, default)
    
    def pop(self, key, default=None):
        """Return the value of the specified key if it exists; otherwise, return the default value and remove the key."""
        value = getattr(self, key, default)
        delattr(self, key)
        return value
    
    def set(self, key, value):
        """Set the value of the specified key."""
        setattr(self, key, value)
