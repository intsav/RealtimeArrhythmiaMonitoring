import numpy as np
import time
import argparse
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import load_model
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

def test_tf_model(X_test, y_test, model):
    predictions, exec_times = [], []
    segments = X_test.astype(np.float32)

    y_test = np.argmax(y_test, axis=1)

    for i in range(segments.shape[0]):
        segment = segments[[i],:,:]
        start_time = time.time()

        pred = model.predict(segment, verbose=0)
        
        end_time = time.time()
        exec_times.append(end_time-start_time)
        predictions.append(pred)

    predictions = np.vstack(predictions)
    y_pred = np.argmax(predictions,1)

    exec_times = np.array(exec_times) * 1000

    cm = confusion_matrix(y_test, y_pred)
    ACC_macro, ACC, TPR, TNR, PPV, f1, kappa = evaluate_metrics(cm, y_test, y_pred, True)

    return y_pred, exec_times, (cm, ACC_macro, ACC, TPR, TNR, PPV, f1, kappa)    

def load_test_data():
    _, _, _, _, X_test, Y_test = load_data()
    
    ohe = OneHotEncoder()
    Y_test = ohe.fit_transform(Y_test.reshape(-1,1))
    
    return X_test, Y_test

def main(model_path) -> object:
    model = load_model(model_path)
    X_test, Y_test = load_test_data()
    
    y_pred_1, exec_times_1, cm = test_tf_model(X_test, Y_test, model)
    print("Mean of execution times: " , np.mean(exec_times_1))
    print("Standard diviation: ", np.std(exec_times_1))
    print("Maximum: ", np.max(exec_times_1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load model')
    parser.add_argument('--model', metavar='path', required=True, help='path to model')
    args = parser.parse_args()    
    
    main(model_path = args.model)