import tensorflow as tf

@tf.function
def train_step(inputs, targets, model, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        total_loss = loss_fn(targets, predictions)

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return total_loss, predictions
