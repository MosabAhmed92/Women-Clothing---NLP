from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred) #scaler

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average= 'macro', labels= [1, 0, -1], zero_division=0) # scalar

def per_class_f1(y_true, y_pred):
    labels = [1, 0, -1]
    label_names = ['Positive', 'Neutral', 'Negative']
    scores = f1_score(y_true, y_pred, average= None, labels= labels, zero_division=0).tolist()
    dict_f1_scores = {}

    for label,lab_name,  score in zip(labels, label_names, scores):
        dic_key = f"{lab_name} ({label})"
        dict_f1_scores[dic_key] = score
    return dict_f1_scores


def confusion(y_true, y_pred):
    return confusion_matrix (y_true, y_pred, labels= [1, 0, -1])  # confusion matrix


def eval_model(y_true, y_pred, model_name):
        
        metrics = {
        "model": model_name,
        "acc_val": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "confusion": confusion(y_true, y_pred),
        "per_class_f1_val": per_class_f1(y_true, y_pred)
        }
        return metrics



def evaluate_model(model, X_tr, y_tr, X_va, y_va, model_name):
    model.fit(X_tr, y_tr)
    ypred_tr = model.predict(X_tr)
    ypred_va = model.predict(X_va)

    metrics = {
        "model": model_name,
        "acc_val": accuracy(y_va, ypred_va),
        "macro_f1_val": macro_f1(y_va, ypred_va),
        "acc_train": accuracy(y_tr, ypred_tr),
        "macro_f1_train": macro_f1(y_tr, ypred_tr),
        "confusion_val": confusion(y_va, ypred_va),
        "per_class_f1_val": per_class_f1(y_va, ypred_va)
    }
    return metrics