from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
import tensorflow as tf
import numpy as np

def evaluate_test_data(test_generator, model, num_classes, config):
    y_true = []
    y_pred = []
    y_pred_probs = []

    # Iterate over the batches of test data
    for batch, (inputs, targets) in enumerate(test_generator):
        predictions = model(inputs, training=False)
        y_true.extend(tf.math.argmax(targets, axis=1))  # Assuming targets are one-hot encoded
        y_pred_probs.extend(predictions.numpy())
        if batch >= test_generator.num_images // config.batch_size:
            break

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate Confusion Matrix and ROC Curve data
    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(num_classes))
    y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes)
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    metrics = {
        "confusion_matrix": cm,
        "class_precision": precision,
        "class_recall": recall,
        "class_f1_score": f1_score,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc
    }


    return metrics, y_true, y_pred, y_pred_probs
