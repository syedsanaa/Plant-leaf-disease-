import torch
import numpy as np
from sam2.build_sam import build_sam2
from keras.layers import Layer, LeakyReLU, add, Conv2D, PReLU, ReLU, Concatenate, Activation, MaxPool2D, Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, ZeroPadding2D
import tensorflow as tf
from tensorflow.keras import backend as K

height = 128
width = 128
in_channels = 3
num_classes = 2

def get_sam2_encoder_output(image):
    """
    Pass an image through SAM2's image encoder and return the output as a NumPy array.
    
    Args:
        image (np.ndarray): Input image (H, W, C) as a NumPy array.

    Returns:
        np.ndarray: Encoded feature map.
    """
    sam2_checkpoint = "/mnt/hdd2/anshul5/imseg-SAM2-UNet/sam2_hiera_tiny.pt"
    model_cfg = "sam2_hiera_t.yaml"

    print("Loading SAM2 model...")
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    print("SAM2 model loaded.")

    sam2_model.sam_mask_decoder.eval()
    sam2_model.sam_prompt_encoder.eval()
    sam2_model.image_encoder.train(True)
    print("SAM2 encoder prepared for training.")

    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()
    print(f"Image tensor shape: {image_tensor.shape}")

    with torch.no_grad():
        print("Passing image through SAM2 encoder...")
        encoder_output = sam2_model.image_encoder.trunk(image_tensor)
        print(f"Encoder output shape: {encoder_output.shape}")

    encoder_output_np = encoder_output.cpu().numpy()
    print(f"Encoder output (NumPy): {encoder_output_np.shape}")
    return encoder_output_np

class Sam2EncoderLayer(Layer):
    def __init__(self, height, width, **kwargs):
        super(Sam2EncoderLayer, self).__init__(**kwargs)
        self.height = height
        self.width = width

    def call(self, inputs):
        """
        This function will be called during the forward pass.
        It will use the SAM2 encoder to process the input image.
        """
        # Define a function to run in tf.py_function
        def encoder_function(image):
            # Convert the TensorFlow tensor to a NumPy array
            image_np = image.numpy()  # Convert tensor to numpy

            # Pass the image through the SAM2 encoder (PyTorch-based)
            sam2_output = get_sam2_encoder_output(image_np)

            # Convert the output back to a TensorFlow tensor
            return tf.convert_to_tensor(sam2_output, dtype=tf.float32)
        # Wrap the PyTorch model call in tf.py_function
        sam2_output = tf.py_function(func=encoder_function, inp=[inputs], Tout=tf.float32)

        # Ensure the output tensor has the correct shape
        #sam2_output.set_shape([None, 128, 128, 64])
        sam2_output.set_shape([None, 128, 128, 4])  # Set the expected shape of the output tensor

        return sam2_output

def planet():
    """
    Create dynamic MNET model object with SAM2 encoder.

    Returns:
        tf.keras.Model: Modified U-Net model with SAM2 encoder.
    """
    no_layer = 0
    inp_size = height # 256 in our case
    start_filter = 4

    while inp_size > 8:
        no_layer += 1
        inp_size = inp_size / 2
        #start_filter *= 2
    #start_filter *= 2

    print("Start Filter: ", start_filter)

    inputs = Input((height, width, in_channels))

    # building model encoder
    encoder = {}
    '''
    inputs = Input((height, width, in_channels))
    for i in range(no_layer):
        if i == 0:
            encoder["enc_{}_0".format(i)] = Conv2D(start_filter, (3, 3), name="enc_{}_0".format(i), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        else:
            encoder["enc_{}_0".format(i)] = Conv2D(start_filter, (3, 3), name="enc_{}_0".format(i), activation='relu', kernel_initializer='he_normal', padding='same')(encoder["mp_{}".format(i-1)])
        start_filter *= 2
        encoder["enc_{}_1".format(i)] = Conv2D(start_filter, (3, 3), name="enc_{}_1".format(i), activation='relu', kernel_initializer='he_normal', padding='same')(encoder["enc_{}_0".format(i)])
        encoder["mp_{}".format(i)] = MaxPooling2D((2, 2), name="mp_{}".format(i))(encoder["enc_{}_1".format(i)])
    '''
    for i in range(no_layer):
        if i == 0:
            # Use SAM2EncoderLayer as the first encoder layer
            encoder["enc_{}_0".format(i)] = Sam2EncoderLayer(height, width, name="sam2_encoder_layer")(inputs)
        else:
            # Add convolutional layers in subsequent encoder stages
            encoder["enc_{}_0".format(i)] = Conv2D(start_filter, (3, 3), name="enc_{}_0".format(i), activation='relu', kernel_initializer='he_normal', padding='same')(encoder["mp_{}".format(i - 1)])
        
        # Add another convolutional layer
        encoder["enc_{}_1".format(i)] = Conv2D(start_filter, (3, 3), name="enc_{}_1".format(i), activation='relu', kernel_initializer='he_normal', padding='same')(encoder["enc_{}_0".format(i)])
        
        # Add max pooling
        encoder["mp_{}".format(i)] = MaxPooling2D((2, 2), name="mp_{}".format(i))(encoder["enc_{}_1".format(i)])
        start_filter *= 2

    '''
    # Use SAM2 encoder to replace original encoder logic
    encoder_output = Sam2EncoderLayer(height, width)(inputs)
    print(height, width, in_channels)
    # Ensure encoder output has a valid shape before resizing
    if encoder_output.shape.is_compatible_with([None, height, width, 64]):
        encoder_resized = tf.image.resize(images=encoder_output, size=(height // 2**no_layer, width // 2**no_layer), method='bilinear')
    else:
        raise ValueError(f"Encoder output shape is not compatible: {encoder_output.shape}")

    '''
    # Use resized encoder output in the decoder
    #mid_1 = Conv2D(start_filter, (3, 3), activation='relu', padding='same')(encoder_resized)
    mid_1 = Conv2D(start_filter, (3, 3), name="mid_1", activation='relu', padding='same')(encoder["mp_{}".format(no_layer-1)])
    start_filter *= 2
    mid_drop = Dropout(0.3)(mid_1)
    mid_2 = Conv2D(start_filter * 2, (3, 3), name="mid_2", activation='relu', padding='same')(mid_drop)

    # Build the decoder (unchanged)
    start_filter = start_filter / 2
    decoder = {}
    for i in range(no_layer):
        if i == 0:
            decoder["dec_T_{}".format(i)] = Conv2DTranspose(start_filter, (2, 2), name="dec_T_{}".format(i), strides=(2, 2), padding='same')(mid_2)
        else:
            decoder["dec_T_{}".format(i)] = Conv2DTranspose(start_filter, (2, 2), name="dec_T_{}".format(i), strides=(2, 2), padding='same')(decoder["dec_{}_1".format(i-1)])

        # Add padding to make the shapes compatible
        enc_shape = K.int_shape(encoder["enc_{}_1".format(no_layer-i-1)])
        dec_shape = K.int_shape(decoder["dec_T_{}".format(i)])
        if enc_shape[1] != dec_shape[1] or enc_shape[2] != dec_shape[2]:
            padding = ((0, enc_shape[1] - dec_shape[1]), (0, enc_shape[2] - dec_shape[2]))
            decoder["dec_T_{}".format(i)] = ZeroPadding2D(padding)(decoder["dec_T_{}".format(i)])
        
        decoder["cc_{}".format(i)] = concatenate([decoder["dec_T_{}".format(i)], encoder["enc_{}_1".format(no_layer-i-1)]], axis=3)
        start_filter = start_filter / 2
        #decoder["dec_{}_0".format(i)] = Conv2D(start_filter, (3, 3), name="dec_{}_0".format(i), activation='relu', padding='same')(decoder["dec_T_{}".format(i)])
        decoder["dec_{}_0".format(i)] = Conv2D(start_filter, (3, 3), name="dec_{}_0".format(i), activation='relu', padding='same')(decoder["cc_{}".format(i)])
        decoder["dec_{}_1".format(i)] = Conv2D(start_filter, (3, 3), name="dec_{}_1".format(i), activation='relu', padding='same')(decoder["dec_{}_0".format(i)])

    outputs = Conv2D(num_classes, (1, 1), activation='softmax', dtype='float32')(decoder["dec_{}_1".format(no_layer-1)])
    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])

    return model

if __name__ == '__main__':
    model = planet()
    model.summary()
    #dummy_input = tf.random.uniform(shape=(1, 256, 256, 3))
    #_ = model(dummy_input)
