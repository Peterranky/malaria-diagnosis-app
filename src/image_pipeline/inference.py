import tensorflow as tf
import numpy as np
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="out_relu"):
    """
    Generates a Grad-CAM heatmap for a given image array and model.
    Compatible with Keras 3 / TensorFlow 2.16+.
    """
    # The base_model (MobileNetV2) is the first layer in our Sequential model
    base_model = model.layers[0]
    
    # Build a sub-model that outputs the last conv layer activations
    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    conv_output_model = tf.keras.models.Model(base_model.input, last_conv_layer.output)
    
    # Run the forward pass manually under GradientTape
    with tf.GradientTape() as tape:
        # Get the conv layer output
        conv_output = conv_output_model(img_array)
        tape.watch(conv_output)
        
        # Run conv output through the rest of the model
        # GlobalAveragePooling2D -> Dense(1, sigmoid)
        x = model.layers[1](conv_output)   # GlobalAveragePooling2D
        preds = model.layers[2](x)          # Dense(1, sigmoid)
        class_channel = preds[:, 0]

    # Gradient of the output neuron w.r.t. the conv layer output
    grads = tape.gradient(class_channel, conv_output)

    # Mean intensity of the gradient over each feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the channels by importance and sum
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    """
    Overlays the heatmap on the original image and saves it.
    """
    # Load the original image
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Create an image with RGB colorized heatmap
    jet = tf.keras.preprocessing.image.array_to_img(jet)
    jet = jet.resize((img.shape[1], img.shape[0]))
    jet = tf.keras.preprocessing.image.img_to_array(jet)

    # Superimpose the heatmap on original image
    superimposed_img = jet * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)
    return cam_path
