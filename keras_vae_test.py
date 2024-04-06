import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import keras.ops as ops


from keras.layers import Dense, Layer
from keras import Model


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init___(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # print(input_shape) # (batch_size, ...)
        self.seed_generator = keras.random.SeedGenerator(41)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch, dim = ops.shape(z_mean)

        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


class Encoder(Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=32, intermediate_dim=64, name="encoder", **kwargs):
        super().__init__(name=name, **kwargs)

        self.dense_proj = Dense(units=intermediate_dim, activation="relu")
        self.dense_mean = Dense(latent_dim)
        self.dense_log_var = Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(Layer):

    def __init__(self, original_dim, intermediate_dim=64, name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense_proj = Dense(intermediate_dim, activation="relu")
        self.dense_output = Dense(original_dim, activation="sigmoid")

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)


class VariationalAutoencoder(Model):

    def __init__(self, original_dim, intermediate_dim=64, latent_dim=32, name="autoencoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim=original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)

        # Add KL divergence regularization loss
        kl_loss = -0.5 * ops.mean(z_log_var - ops.square(z_mean) - ops.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed


if __name__ == "__main__":
    (x_train, _), _ = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype("float32") / 255.0

    original_dim = 784
    vae = VariationalAutoencoder(original_dim=original_dim)
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    vae.compile(
        optimizer,
        loss=keras.losses.MeanSquaredError(),
        metrics=[
            "accuracy",
        ],
    )

    vae.fit(x_train, x_train, epochs=2, batch_size=64)
