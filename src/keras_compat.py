"""
src/keras_compat.py
===================
Cross-version compatibility shim for Keras 3 model loading.
Resolves unexpected keyword arguments ('input_axes', 'output_axes') when deserializing
initializers saved across different Keras 3 minor releases.
"""

def patch_keras_initializers() -> None:
    """Safely patches Keras Initializer subclasses to ignore deprecated/new serialization kwargs."""
    try:
        import keras
        import keras.initializers
        for cls in list(vars(keras.initializers).values()):
            if isinstance(cls, type) and issubclass(cls, keras.initializers.Initializer):
                orig_init = cls.__init__
                def _make_patched(orig):
                    return lambda self, *a, **kw: orig(
                        self, *a, **{k: v for k, v in kw.items() if k not in ("input_axes", "output_axes")}
                    )
                cls.__init__ = _make_patched(orig_init)
    except Exception:
        pass

# Automatically patch on import
patch_keras_initializers()
