# # ml_model/model.py


import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_image_model(input_shape, num_classes):
    """
    Creates a high-accuracy image classification model using MobileNetV2 transfer learning.
    """
    # 1. Load the pre-trained MobileNetV2 model, excluding its final classification layer.
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,  # We don't need the original final layer
        weights='imagenet'
    )

    # 2. Freeze the base model's layers
    # This prevents its learned weights from being updated during our training.
    base_model.trainable = False

    # 3. Add our custom classifier on top
    x = base_model.output
    x = Flatten()(x)  # Flatten the features to a 1D vector
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)  # Add dropout for regularization
    # The final output layer has neurons equal to the number of classes (letters).
    predictions = Dense(num_classes, activation='softmax')(x)

    # 4. Construct the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

if __name__ == '__main__':
    # --- Configuration ---
    IMAGE_SIZE = (64, 64)
    BATCH_SIZE = 32
    DATA_PATH = 'data/asl_alphabet_train' # Path to your ASL letter images

    # --- Data Loading and Augmentation ---
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2 
    )

    train_generator = train_datagen.flow_from_directory(
        DATA_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        DATA_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # --- Model Creation and Compilation ---
    num_classes = train_generator.num_classes
    model = create_image_model(input_shape=(*IMAGE_SIZE, 3), num_classes=num_classes)
    
    # We use a lower learning rate for transfer learning to fine-tune effectively.
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # --- Train the Model ---
    history = model.fit(
        train_generator,
        epochs=25,
        validation_data=validation_generator
    )

    # --- Save the Model ---
    model.save('trained_model/asl_image_model.h5')
    print("Image-based model trained and saved successfully!")
