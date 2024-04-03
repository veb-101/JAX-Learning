# Local script: Minor difference in data loading process as done in JAX_MLP_MNIST_SMPD_torchvision.ipynb
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"  # Use 8 CPU

import time
from functools import partial

import numpy as np
import numpy.random as npr

import jax
import jax.random as random
from jax import jit, vmap, pmap, grad, value_and_grad
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map
import jax.numpy as jnp
from jax import lax

import mnist_dataset_download as datasets
from typing import List, Tuple, Dict


def init_random_params(layer_widths: List, parent_key, scale: float = 0.01):

    params = []

    keys = random.split(parent_key, num=len(layer_widths) - 1)

    for num_in_nodes, num_out_nodes, key in zip(layer_widths[:-1], layer_widths[1:], keys):
        weight_key, bias_key = random.split(key)

        params.append(
            [
                scale * random.normal(weight_key, shape=(num_in_nodes, num_out_nodes)),
                scale * random.normal(bias_key, shape=(num_out_nodes,)),
            ]
        )

    return params


def predict(params, x):

    hidden_layers = params[:-1]

    activation = x

    for w, b in hidden_layers:
        activation = jax.nn.relu(jnp.dot(activation, w) + b)  # 16x784 x 512x784 = 16x784 x 784x512 = 16x512 x 512x256 = 16x256 x 256x10 = 16x10

    w_last, b_last = params[-1]

    logits = jnp.dot(activation, w_last) + b_last
    # return logits - logsumexp(logits, axis=0)
    return logits - logsumexp(logits, axis=1, keepdims=True)


def loss(params, batch):
    images, labels = batch
    # print("images.shape, labels.shape", images.shape, labels.shape)
    prediction = predict(params, images)
    # print("prediction.shape", prediction.shape)
    return -jnp.mean(jnp.sum(prediction * labels, axis=1))


@jit
def accuracy(params, batch):
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(predict(params, inputs), axis=1)
    return jnp.mean(predicted_class == target_class)


@partial(pmap, axis_name="batch", static_broadcasted_argnums=2)
def spmd_update(params, batch, step_size):
    # grads = grad(loss)(params, batch)
    loss_value, grads = value_and_grad(loss)(params, batch)

    # We compute the total gradients, summing across the device-mapped axis,
    # using the `lax.psum` SPMD primitive, which does a fast all-reduce-sum.
    grads = [(lax.pmean(dw, "batch"), lax.pmean(db, "batch")) for dw, db in grads]

    """ 
    When you use pmap in JAX, it parallelizes the function execution across multiple devices, and the result you get back is a "sharded" array where each element corresponds to the result from one of the devices. Even if you aggregate the values using lax.pmean, when you print the result outside the pmap function, you will see an array where each element is the same aggregated value, one from each device.
    """
    loss_value_aggr = lax.pmean(loss_value, axis_name="batch")

    return loss_value_aggr, [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(params, grads)]


def get_number_of_batches(dataset_size, batch_size):
    dataset_length = dataset_size.shape[0]
    num_complete_batches, leftover = divmod(dataset_length, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    return num_batches


def data_stream(train_images, train_labels, batch_size, num_batches, num_devices):
    rng = npr.RandomState(0)

    while True:
        perm = rng.permutation(train_images.shape[0])
        for i in range(num_batches):
            batch_idx = perm[i * batch_size : (i + 1) * batch_size]
            images, labels = train_images[batch_idx], train_labels[batch_idx]
            # For this SPMD example, we reshape the data batch dimension into two
            # batch dimensions, one of which is mapped over parallel devices.
            batch_size_per_device, ragged = divmod(images.shape[0], num_devices)
            if ragged:
                msg = "batch size must be divisible by device count, got {} and {}."
                raise ValueError(msg.format(batch_size, num_devices))
            shape_prefix = (num_devices, batch_size_per_device)
            images = images.reshape(shape_prefix + images.shape[1:])
            labels = labels.reshape(shape_prefix + labels.shape[1:])
            yield images, labels


if __name__ == "__main__":
    SEED = 42
    MNIST_IMG_SIZE = (28, 28)
    NUM_INP_NODES = np.prod(MNIST_IMG_SIZE)
    NUM_CLASSES = 10

    layer_sizes = [NUM_INP_NODES, 512, 256, NUM_CLASSES]
    param_scale = 0.1
    step_size = 0.001
    num_epochs = 3
    batch_size = 128

    train_images, train_labels, test_images, test_labels = datasets.mnist()

    num_devices = jax.device_count()
    num_batches = get_number_of_batches(train_images, batch_size)

    batches = data_stream(train_images, train_labels, batch_size, num_batches, num_devices)

    # We replicate the parameters so that the constituent arrays have a leading
    # dimension of size equal to the number of devices we're pmapping over.
    key = random.PRNGKey(SEED)
    init_params = init_random_params(layer_sizes, key, param_scale)
    replicate_array = lambda x: np.broadcast_to(x, (num_devices,) + x.shape)
    replicated_params = tree_map(replicate_array, init_params)

    for epoch in range(num_epochs):
        start_time = time.time()
        for cnt in range(num_batches):
            loss_value_b, replicated_params = spmd_update(replicated_params, next(batches), step_size)
            if cnt % 50 == 0:

                # Convert the sharded array to a regular NumPy array and take the first element
                # Since loss_value_aggr is the same across all devices, taking the first element is sufficient
                final_loss_value = loss_value_b[0]
                print("loss_value_b", final_loss_value)
        epoch_time = time.time() - start_time

        # We evaluate using the jitted `accuracy` function (not using pmap) by
        # grabbing just one of the replicated parameter values.
        params = tree_map(lambda x: x[0], replicated_params)
        train_acc = accuracy(params, (train_images, train_labels))
        test_acc = accuracy(params, (test_images, test_labels))
        print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
        print(f"Training set accuracy {train_acc}")
        print(f"Test set accuracy {test_acc}")
