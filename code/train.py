from model_ecg import model_ecg
from load import load_data
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

def main() -> object:
    filepath = 'models/'
    
    model = model_ecg()
    X_train, Y_train, X_val, Y_val, _, _ = load_data()

    callbacks = [EarlyStopping(monitor="val_loss", patience=10, verbose=1),
                ModelCheckpoint(filepath = filepath +'.{val_loss:.3f}-{val_sparse_categorical_accuracy:.3f}-{epoch:03d}-{loss:.3f}-{sparse_categorical_accuracy:.3f}.h5', 
                                monitor='val_loss', save_best_only=False, verbose=0, period=1)]
    
    model.fit( X_train, Y_train, 
            validation_data = (X_val, Y_val),
            batch_size = 256,
            epochs = 100, 
            callbacks = callbacks
            )

if __name__ == '__main__':
    main()