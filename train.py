import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import custom modules
from src.data_loader import load_mitbih_data
from src.preprocessing import apply_grunwald_letnikov, apply_caputo
from src.models import build_fusion_model

def main():
    # 1. Load the data
    X_train_raw, y_train, X_test_raw, y_test = load_mitbih_data()
    
    # 2. Apply Fractional Calculus (Feature Engineering)
    print("Applying Grünwald-Letnikov and Caputo fractional derivatives...")
    X_train_gl = apply_grunwald_letnikov(X_train_raw)
    X_test_gl = apply_grunwald_letnikov(X_test_raw)
    
    X_train_cap = apply_caputo(X_train_raw)
    X_test_cap = apply_caputo(X_test_raw)
    
    # 3. Channel Fusion (Raw + GL + Caputo)
    print("Fusing channels...")
    X_train_fusion = np.stack([X_train_raw, X_train_gl, X_train_cap], axis=-1)
    X_test_fusion = np.stack([X_test_raw, X_test_gl, X_test_cap], axis=-1)
    
    # 4. Build the Model
    model = build_fusion_model(input_shape=(187, 3), num_classes=5)
    model.summary()
    
    # 5. Define Optimization Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ]
    
    # 6. Train the Model
    print("Starting training process...")
    history = model.fit(
        X_train_fusion, y_train,
        validation_data=(X_test_fusion, y_test),
        epochs=10, 
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )
    
    # 7. Save the final weights
    model.save("optimized_fusion_model.keras")
    print("Model saved successfully as 'optimized_fusion_model.keras'")

if __name__ == "__main__":
    main()
