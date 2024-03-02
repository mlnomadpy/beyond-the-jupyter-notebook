import argparse
import os
import tensorflow as tf
from utils.build_fusion_model import build_model
from utils.load_test_set import load_test_data
from utils.evaluate_test_data import evaluate_test_data
import matplotlib.pyplot as plt
import seaborn as sns
from utils.setup_gpus import setup_gpus
from utils.set_seed import set_seed
from utils.load_data_multimodal import load_data
import wandb
import numpy as np


def test(config):
    model_name = 'BasicCNN'


    try:
        strategy = tf.distribute.MirroredStrategy() if len(config.gpu.split(',')) > 1 else tf.distribute.get_strategy()
    except ValueError as e:
        print(f"Error setting up GPU strategy: {e}")
        strategy = tf.distribute.get_strategy()

        print("Number of accelerators: ", strategy.num_replicas_in_sync)

    wandb.login()

    run = wandb.init(
        project="SeeNN_exp_test",
        entity="rowanvisibility",
        name=model_name
    )

    wandb.config.update(vars(config))


    print("Loading the Model...")
    with strategy.scope():
        model = build_model(config)
        model.summary()

    model.load_weights(config.model_path)

    print("Loading Test Data...")
    _ , val_data = load_data(config)

    print("Evaluating Model on Test Dataset...")
    metrics, y_true, y_pred, y_pred_probs = evaluate_test_data(val_data, model, config.num_classes, config)


    class_names = [f'bin_{i}' for i in range(config.num_classes)]
    print("Saving Results...")

    # Display results
    # Plot and save confusion matrix
    # Example confusion matrix
    confusion_matrix = np.array(metrics['confusion_matrix'])

    # Calculate the percentages
    matrix_percentages = confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis]
    matrix_percentages = np.nan_to_num(matrix_percentages)  # Replace NaNs with zeros


    plt.figure(figsize=(10, 8))

    # Plot the percentages in the heatmap
    sns.heatmap(matrix_percentages, annot=True, fmt=".2%", cmap='Blues', 
                annot_kws={"size": 16}, cbar_kws={'shrink': .5}, 
                xticklabels=True, yticklabels=True)

    # Define colors for text
    off_diagonal_color = "black"
    diagonal_color = "white"

    # Overlay the counts, changing the color for the diagonal
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            text = f'{confusion_matrix[i, j]}'
            color = diagonal_color if i == j else off_diagonal_color
            plt.text(j+0.5, i+0.4, text, 
                    horizontalalignment='center', verticalalignment='bottom', 
                    color=color, fontsize=16)

    plt.title('SeeNN - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()  # Adjust layout to fit title and axis labels
    cm_path = os.path.join(wandb.run.dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()

    # Plot and save ROC curve
    plt.figure()
    for i in range(config.num_classes):
        plt.plot(metrics["fpr"][i], metrics["tpr"][i], label='Class {} (area = {:.2f})'.format(i, metrics["roc_auc"][i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    roc_path = os.path.join(wandb.run.dir, 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()


    print("Testing completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a CNN for image classification.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of Classes')
    parser.add_argument('--img_height', type=int, default=224, help='Image height.')
    parser.add_argument('--img_width', type=int, default=224, help='Image width.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--gpu', type=str, default='2', help='GPU device to use.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing.')
    parser.add_argument('--num_img_lim', type=int, default=100000, help='Number of images per class to load for testing.')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation Split')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset directory.')

    args = parser.parse_args()
    # Set seed and GPU configuration
    set_seed(args.seed)
    setup_gpus(args.gpu)
    test(args)


