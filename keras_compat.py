# keras_compat.py
# Aggressively patch Keras preprocessing layers to ignore 'value_range'
import logging

def patch_keras_preprocessing():
    try:
        from keras.src.layers.preprocessing.image_preprocessing import (
            random_rotation,
            random_translation,
            random_zoom,
            random_contrast,
        )

        layers_to_patch = [
            (random_rotation, "RandomRotation"),
            (random_translation, "RandomTranslation"),
            (random_zoom, "RandomZoom"),
            (random_contrast, "RandomContrast"),
        ]

        for module, class_name in layers_to_patch:
            if not hasattr(module, class_name):
                continue
            cls = getattr(module, class_name)
            
            # Patch the __init__ of the EXISTING class
            orig_init = cls.__init__
            if hasattr(orig_init, "_is_patched"):
                continue

            def make_patched_init(old_init):
                def patched_init(self, *args, **kwargs):
                    kwargs.pop("value_range", None)
                    return old_init(self, *args, **kwargs)
                patched_init._is_patched = True
                return patched_init

            cls.__init__ = make_patched_init(orig_init)

        # Also ensure they are registered for Keras 3 deserialization
        from keras import layers
        for _, class_name in layers_to_patch:
            if hasattr(layers, class_name):
                cls = getattr(layers, class_name)
                # Just in case the reference is different
                orig_init = cls.__init__
                if not hasattr(orig_init, "_is_patched"):
                    cls.__init__ = make_patched_init(orig_init)

        logging.info("Patched Keras preprocessing layers via keras_compat.")
    except Exception as e:
        logging.warning(f"Failed to apply Keras patches in keras_compat: {e}")

# Run on import
patch_keras_preprocessing()
