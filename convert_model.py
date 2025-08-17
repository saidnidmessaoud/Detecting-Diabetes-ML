import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D

# Custom layer to handle compatibility
class FixedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)  # Remove problematic parameter
        super().__init__(*args, **kwargs)

# Load and convert the model
try:
    model = tf.keras.models.load_model(
        "classifier.h5",
        custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D}
    )
    model.save("converted_classifier", save_format="tf")
    print("Model converted successfully to SavedModel format!")
except Exception as e:
    print(f"Conversion failed: {str(e)}")