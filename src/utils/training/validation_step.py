import tensorflow as tf
@tf.function
def val_step(inputs, targets, model, loss_fn):
    predictions = model(inputs, training=False)
    total_v_loss = loss_fn(targets, predictions)
    return total_v_loss, predictions
