import os

# This guide can only be run with the jax backend.
os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras
import numpy as np
import mnist_dataset_download as datasets


def get_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x1 = keras.layers.Dense(64, activation="relu")(inputs)
    x2 = keras.layers.Dense(64, activation="relu")(x1)
    outputs = keras.layers.Dense(10, name="predictions")(x2)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def compute_loss_and_updates(trainable_variables, non_trainable_variables, metric_variables, x, y):
    y_pred, non_trainable_variables = model.stateless_call(trainable_variables, non_trainable_variables, x)
    loss = loss_fn(y, y_pred)

    metric_variables = train_acc_metric.stateless_update_state(metric_variables, y, y_pred)
    return loss, (non_trainable_variables, metric_variables)


@jax.jit
def train_step(state, data):
    (trainable_variables, non_trainable_variables, optimizer_variables, metric_variables) = state
    x, y = data

    (loss, (non_trainable_variables, metric_variables)), grads = grad_fn(
        trainable_variables,
        non_trainable_variables,
        metric_variables,
        x,
        y,
    )

    trainable_variables, optimizer_variables = optimizer.stateless_apply(
        optimizer_variables,
        grads,
        trainable_variables,
    )

    # Return updated state
    return loss, (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
        metric_variables,
    )


@jax.jit
def eval_step(state, data):
    trainable_variables, non_trainable_variables, metric_variables = state
    x, y = data

    y_pred, non_trainable_variables = model.stateless_call(trainable_variables, non_trainable_variables, x)

    loss = loss_fn(y, y_pred)

    metric_variables = val_acc_metric.stateless_update_state(metric_variables, y, y_pred)

    return loss, (
        trainable_variables,
        non_trainable_variables,
        metric_variables,
    )


def get_number_of_batches(dataset_size, batch_size):
    dataset_length = dataset_size.shape[0]
    num_complete_batches, leftover = divmod(dataset_length, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    return num_batches


def data_stream(train_images, train_labels, batch_size, num_batches, num_devices):
    key = jax.random.PRNGKey(0)

    while True:
        perm = jax.random.permutation(key, x=train_images.shape[0])
        for i in range(num_batches):
            batch_idx = perm[i * batch_size : (i + 1) * batch_size]
            images, labels = train_images[batch_idx], train_labels[batch_idx]
            # For this SPMD example, we reshape the data batch dimension into two
            # batch dimensions, one of which is mapped over parallel devices.
            batch_size_per_device, ragged = divmod(images.shape[0], num_devices)
            if ragged:
                msg = "batch size must be divisible by device count, got {} and {}."
                raise ValueError(msg.format(batch_size, num_devices))
            # shape_prefix = (num_devices, batch_size_per_device)
            # images = images.reshape(shape_prefix + images.shape[1:])
            # labels = labels.reshape(shape_prefix + labels.shape[1:])

            images = images.reshape(batch_size_per_device, -1)
            labels = labels.reshape(batch_size_per_device, -1)

            yield images, labels


if __name__ == "__main__":
    NUM_EPOCHS = 2
    BATCH_SIZE = 32
    model = get_model()

    train_images, train_labels, test_images, test_labels = datasets.mnist()

    num_devices = jax.device_count()

    num_train_batches = get_number_of_batches(train_images, BATCH_SIZE)
    train_dataset = data_stream(train_images, train_labels, BATCH_SIZE, num_train_batches, num_devices)

    num_test_batches = get_number_of_batches(test_images, BATCH_SIZE)
    test_dataset = data_stream(test_images, test_labels, BATCH_SIZE, num_test_batches, num_devices)

    for img_batch, lab_batch in train_dataset:
        print(img_batch.shape, lab_batch.shape)
        break

    # Instantiate a loss function.
    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

    grad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)

    # Prepare the metrics.
    train_acc_metric = keras.metrics.CategoricalAccuracy()
    val_acc_metric = keras.metrics.CategoricalAccuracy()

    # Instantiate an optimizer.
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    # Build optimizer variables.
    optimizer.build(model.trainable_variables)

    trainable_variables = model.trainable_variables
    non_trainable_variables = model.non_trainable_variables
    optimizer_variables = optimizer.variables
    metric_variables = train_acc_metric.variables

    state = (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
        metric_variables,
    )

    # Training loop
    for epoch in range(NUM_EPOCHS):

        for step in range(num_train_batches):
            loss, state = train_step(state, next(train_dataset))

            # Log every 100 batches.
            if step % 100 == 0:
                print(f"Epoch {epoch} :: Training loss (for 1 batch) at step {step}: {float(loss):.4f}")
                _, _, _, metric_variables = state
                for variable, value in zip(train_acc_metric.variables, metric_variables):
                    variable.assign(value)
                print(f"Training accuracy: {train_acc_metric.result()}")
                print(f"Seen so far: {(step + 1) * BATCH_SIZE} samples")

        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            _,
        ) = state

        val_metric_variables = val_acc_metric.variables
        val_state = trainable_variables, non_trainable_variables, val_metric_variables

        # Eval loop
        for step in range(num_test_batches):
            loss, val_state = eval_step(val_state, next(test_dataset))

            if step % 100 == 0:
                print(f"Validation loss (for 1 batch) at step {step}: {float(loss):.4f}")
                _, _, val_metric_variables = val_state
                for variable, value in zip(val_acc_metric.variables, val_metric_variables):
                    variable.assign(value)
                print(f"Validation accuracy: {val_acc_metric.result()}")
                print(f"Seen so far: {(step + 1) * BATCH_SIZE} samples")

        print("--------------------------------------------")
