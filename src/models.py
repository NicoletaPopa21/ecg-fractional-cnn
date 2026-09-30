import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.models import Model

def build_base_model(input_shape=(187, 1), num_classes=5):
    """
    Builds the baseline 1D CNN model for single-channel input (raw ECG signal).
    """
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 7, padding='same', activation='relu')(inputs)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 5, padding='same', activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name="Baseline_Model")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def build_fusion_model(input_shape=(187, 3), num_classes=5):
    """
    Builds the optimized 1D CNN model for 3-channel input (Raw + GL + Caputo).
    Includes Dropout layer for regularization and preventing overfitting.
    """
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 7, padding='same', activation='relu')(inputs)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 5, padding='same', activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    
    # Dropout layer for regularization
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name="Optimized_Fusion_Model")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
