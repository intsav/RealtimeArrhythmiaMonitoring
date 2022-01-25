
import numpy as np
import time
import tensorflow as tf
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix
from tensorflow.keras.models import load_model
import tensorflow.lite as tflite
from load import load_data

def evaluate_metrics(confusion_matrix, y_test, y_pred, print_result=False, f1_avg='macro'):
    
    TP = np.diag(confusion_matrix)
    FP = confusion_matrix.sum(axis=0) - TP
    FN = confusion_matrix.sum(axis=1) - TP    
    TN = confusion_matrix.sum() - (FP + FN + TP)
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    PPV = TP / (TP + FP)
    ACC = (TP + TN) / (TP + FP + FN + TN)

    ACC_macro = np.mean(ACC)

    f1 = f1_score(y_test, y_pred, average=f1_avg)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    if (print_result):
        print("\n")
        print("\n")
        print("============ METRICS ============")
        print(confusion_matrix)
        print("Accuracy (macro) : ", ACC_macro)        
        print("F1 score         : ", f1)
        print("Cohen Kappa score: ", kappa)
        print("======= Per class metrics =======")
        print("Accuracy         : ", ACC)
        print("Sensitivity (TPR): ", TPR)
        print("Specificity (TNR): ", TNR)
        print("Precision (+P)   : ", PPV)
    
    return ACC_macro, ACC, TPR, TNR, PPV, f1, kappa


def load_lite_model(lite_model=None, tflite_load='Loaded', filename=None):
    if tflite_load == 'Loaded':
        interpreter = tflite.Interpreter(model_content=lite_model)
    elif tflite_load == 'File':
        interpreter = tflite.Interpreter(model_path=filename + ".tflite")

    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    return interpreter, input_index, output_index


def load_test_data():
    _, _, _, _, X_test, Y_test = load_data()
    return X_test, Y_test


def test_lite_model(X_test, y_test, interpreter, input_index, output_index):
    predictions = []
    exec_times  = []
    segments = X_test.astype(np.float32)

    y_test = np.argmax(y_test, axis=1)

    for i in range(segments.shape[0]):
        segment = segments[[i],:,:]
        start_time = time.time()
        interpreter.set_tensor(input_index, segment)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_index)
        end_time = time.time()
        exec_times.append(end_time-start_time)
        predictions.append(pred)

    predictions = np.vstack(predictions)
    y_pred = np.argmax(predictions,1)

    exec_times = np.array(exec_times) * 1000

    cm = confusion_matrix(y_test, y_pred)
    ACC_macro, ACC, TPR, TNR, PPV, f1, kappa = evaluate_metrics(cm, y_test, y_pred, True)

    return y_pred, exec_times, (cm, ACC_macro, ACC, TPR, TNR, PPV, f1, kappa) 

def main() -> object:
    X_test, Y_test = load_test_data()

    FILENAME = 'models/ecg_quant'
    interpreter_1, input_idx_1, output_idx_1 = load_lite_model(tflite_load='File', filename=FILENAME)
    y_pred, exec_times, cm = test_lite_model(X_test, Y_test, interpreter_1, input_idx_1, output_idx_1)

    print("Mean of execution times: " , np.mean(exec_times))
    print("Standard diviation: ", np.std(exec_times))
    print("Maximum: ", np.max(exec_times))

if __name__ == '__main__':
    main()