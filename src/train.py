#!/usr/bin/env python3

"""
Author: Taha Bouhsine
Email: contact@tahabouhsine.com
Created on: March 01st 2024
Last Modified: March 01st 2024
Description:
This Python script (train.py) is designed for training an image classification model. 
It includes functionality for loading data, setting up a model, and training and evaluating it.
"""

import argparse
import tensorflow as tf
from wandb.keras import WandbCallback
from tensorflow.keras.optimizers import Adam

import os
import numpy as np
import wandb

import matplotlib.pyplot as plt
import seaborn as sns

from utils.training.train_step import train_step
from utils.training.validation_step import val_step

from utils.testing.evaluate_test_data import evaluate_test_data

from utils.data_loaders.load_data import load_data

from utils.models.build_model import build_model

from utils.miscs.setup_gpu import setup_gpus
from utils.miscs.set_seeds import set_seed



def train(config):

    model_name = 'BasicCNN'


    num_classes = config.num_classes
    class_names = [f'bin_{i}' for i in range(num_classes)]
    config.class_names = class_names


    try:
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = tf.distribute.get_strategy()
        print("Number of accelerators: ", strategy.num_replicas_in_sync)
    except ValueError as e:
        print(f"Error setting up GPU strategy: {e}")
        strategy = tf.distribute.get_strategy()

    wandb.login()

    run = wandb.init(
        project=config.project,
        entity=config.entity,
        name=config.model_name
    )

    wandb.config.update(vars(config))

    print('Loading the Dataset...')
    # Load data
    train_data, val_data = load_data(config)

    print('Classes: ' + str(class_names))



    # Build model
    print('Building Model:')

    with strategy.scope():
        model = build_model(config)
        model.summary()
        loss_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        # Metrics to measure loss and accuracy for training
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.CategoricalAccuracy(
            name='train_accuracy')
        train_precision = tf.keras.metrics.Precision(name='train_precision')
        train_recall = tf.keras.metrics.Recall(name='train_recall')

        # Metrics to measure loss and accuracy for validation
        val_loss = tf.keras.metrics.Mean(name='val_loss')
        val_accuracy = tf.keras.metrics.CategoricalAccuracy(
            name='val_accuracy')
        val_precision = tf.keras.metrics.Precision(name='val_precision')
        val_recall = tf.keras.metrics.Recall(name='val_recall')
        optimizer = Adam()

        # Initialize metrics
        class_accuracy = [tf.keras.metrics.BinaryAccuracy(
            name=f'class_{i}_accuracy') for i in range(5)]
        class_precision = [tf.keras.metrics.Precision(
            name=f'class_{i}_precision') for i in range(5)]
        class_recall = [tf.keras.metrics.Recall(
            name=f'class_{i}_recall') for i in range(5)]
    wandb_callback = WandbCallback(save_model=False)

    wandb_callback.set_model(model)
    
    best_val_loss = 10000
    # Train for a few epochs
    print("Training Start")
    for epoch in range(config.epochs):
        print(f'Epoch {epoch}:')
        # Reset the metrics at start of each epoch
        train_accuracy.reset_states()
        train_precision.reset_states()
        train_recall.reset_states()
        val_accuracy.reset_states()
        val_precision.reset_states()
        val_recall.reset_states()
        for class_id in range(config.num_classes):
            class_accuracy[class_id].reset_states()
            class_precision[class_id].reset_states()
            class_recall[class_id].reset_states()

        # Before each epoch, we call the 'on_epoch_begin' method of our callback
        wandb_callback.on_epoch_begin(epoch)
        if epoch == config.trainable_epochs:
            model.trainable = True

        print('Training...')
        # Training loop
        for steps, (inputs, targets) in enumerate(train_data):
            # TODO: augment the inputs with random flip horizental to all the modalities
            total_loss, predictions = train_step(inputs, targets, model, loss_fn, optimizer)
            train_loss.update_state(total_loss)
            train_accuracy.update_state(targets, predictions)
            train_precision.update_state(targets, predictions)
            train_recall.update_state(targets, predictions)
            # Break the loop once we reach the number of steps per epoch
            if steps >= len(train_data.df) // config.batch_size:
                break

        train_data.reset()  # Reset the generator after each epoch

        print('Validation...')
        # # Validation loop
        for steps, (inputs, targets) in enumerate(val_data):
            total_v_loss, predictions = val_step(inputs, targets, model, loss_fn)
            val_loss.update_state(total_v_loss)
            val_accuracy.update_state(targets, predictions)
            val_precision.update_state(targets, predictions)
            val_recall.update_state(targets, predictions)
            
            # Update metrics for each class
            for class_id in range(config.num_classes):
                y_true_class = tf.equal(tf.argmax(targets, axis=-1), class_id)
                y_pred_class = tf.equal(tf.argmax(predictions, axis=-1), class_id)
                
                class_accuracy[class_id].update_state(y_true_class, y_pred_class)
                class_precision[class_id].update_state(y_true_class, y_pred_class)
                class_recall[class_id].update_state(y_true_class, y_pred_class)
                
            if steps == len(val_data.df) // config.batch_size:
                break

        val_data.reset()  # Reset the generator after each epoch

        # Log metrics with Wandb
        logs = {
            'epoch': epoch,
            'train_loss': train_loss.result(),
            'train_accuracy': train_accuracy.result(),
            'train_precision': train_precision.result(),
            'train_recall': train_recall.result(),
            'val_loss': val_loss.result(),
            'val_accuracy': val_accuracy.result(),
            'val_precision': val_precision.result(),
            'val_recall': val_recall.result()
        }

        for class_id in range(config.num_classes):
            logs[f'class_{class_id}_accuracy'] = class_accuracy[class_id].result(
            ).numpy()
            logs[f'class_{class_id}_precision'] = class_precision[class_id].result(
            ).numpy()
            logs[f'class_{class_id}_recall'] = class_recall[class_id].result(
            ).numpy()

        # # # After each epoch, we call the 'on_epoch_end' method of our callback
        wandb_callback.on_epoch_end(epoch, logs=logs)

        print(f'Epoch {epoch + 1}, '
              f'Train Loss: {train_loss.result().numpy()}, '
              f'Train Accuracy: {train_accuracy.result().numpy()}, '
              f'Val Loss: {val_loss.result().numpy()}, '
              f'Val Accuracy: {val_accuracy.result().numpy()}'
              )

        for class_id in range(config.num_classes):
            print(f'  Class {class_id}: '
                  f'Accuracy: {class_accuracy[class_id].result().numpy()}, '
                  f'Precision: {class_precision[class_id].result().numpy()}, '
                  f'Recall: {class_recall[class_id].result().numpy()}')

        current_val_loss = val_loss.result().numpy()
        if current_val_loss < best_val_loss:
            model.save(os.path.join(wandb.run.dir, f'{model_name}_best_val_loss_model.keras'))
            best_val_loss = current_val_loss

    print('Saving Last Model')
    # Save the trained model
    model.save(os.path.join(wandb.run.dir, f"last_{model_name}_model.keras"))

    
    print('Evaluating Model on Test Dataset...')

    metrics, y_true, y_pred, y_pred_probs = evaluate_test_data(val_data, model, config.num_classes, config)

    # Log confusion matrix and class-wise metrics to wandb
    class_names = [f'bin_{i}' for i in range(config.num_classes)]
    wandb.log({
        "conf_mat": wandb.plot.confusion_matrix(
            probs=None,
            preds=y_pred,
            y_true=y_true,  # Ensure y_true is defined and accessible here
            class_names=class_names
        ),
        "roc_curve": wandb.plot.roc_curve(y_true, y_pred_probs, labels=class_names)
    })

    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d")
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
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


    print("Training completed successfully!")


if __name__ == '__main__':
    # Parsing command-line arguments
    parser = argparse.ArgumentParser(description='Train a CNN for image classification.')
    parser.add_argument('--img_width', type=int, default=64, help='Image width.')
    parser.add_argument('--img_height', type=int, default=64, help='Image height.')

    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')

    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--test_dataset_path', type=str, required=True, help='Path to the test dataset directory.')
    parser.add_argument('--num_img_lim', type=int, required=True, help='The number of images per class')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of Classes')
    parser.add_argument('--val_split', type=float, required=True, help='Validation Split')


    parser.add_argument('--seed', type=int, default=64, help='Random Seed.')
    parser.add_argument('--gpu', type=str, default='2,3', help='GPUs.')
    parser.add_argument('--trainable_epochs', type=int, required=True, help='The number of epochs before the backbone become trainable')
    parser.add_argument('--model_name', type=str, required=True, help='Model Name.')
    parser.add_argument('--project', type=str, required=True, help='Wandb Project.')
    parser.add_argument('--entity', type=str, required=True, help='Wandb Entity.')

    args = parser.parse_args()



    # Setup GPUs
    setup_gpus(args.gpu)
    set_seed(args.seed)

    train(args)