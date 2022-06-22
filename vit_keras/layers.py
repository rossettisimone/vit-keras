# pylint: disable=arguments-differ,missing-function-docstring,missing-class-docstring,unexpected-keyword-arg,no-value-for-parameter
import tensorflow as tf
import tensorflow_addons as tfa
import typing

ImageSizeArg = typing.Union[typing.Tuple[int, int], int]

@tf.keras.utils.register_keras_serializable()
class ClassToken(tf.keras.layers.Layer):
    """Append a class token to an input layer."""

    def build(self, input_shape):
        cls_init = tf.zeros_initializer()
        self.hidden_size = input_shape[-1]
        self.cls = tf.Variable(
            name="cls",
            initial_value=cls_init(shape=(1, 1, self.hidden_size), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
            dtype=inputs.dtype,
        )
        return tf.concat([cls_broadcasted, inputs], 1)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class AddPositionEmbs(tf.keras.layers.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""
    def __init__(self, image_size: ImageSizeArg, patch_size: int, hidden_size: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.shape=(1, 1 + image_size[0]//patch_size*image_size[1]//patch_size, hidden_size)

    def build(self, input_shape):
        assert (
            len(input_shape) == 3
        ), f"Number of dimensions should be 3, got {len(input_shape)}"
        self.pe = tf.Variable(
            name="pos_embedding",
            initial_value=tf.random_normal_initializer(stddev=0.06)(
                shape=self.shape
            ),
            dtype="float32",
            trainable=True,
        )

    def call(self, inputs):
        return inputs + tf.cast(self.interpolate_pos_encoding(inputs), dtype=inputs.dtype)

    def interpolate_pos_encoding(self, x):
        M = tf.shape(x)[1] - 1
        N = tf.shape(self.pe)[1] - 1

        def interpolate():
            class_pos_embed = self.pe[:, :1]
            patch_pos_embed = self.pe[:, 1:]
            dim = tf.shape(x)[-1]
            MM = tf.cast(tf.math.sqrt(tf.cast(M,tf.float32)),tf.int32)
            NN = tf.cast(tf.math.sqrt(tf.cast(N,tf.float32)),tf.int32)
            patch_pos_embed = tf.image.resize(
                tf.reshape(patch_pos_embed,(1, NN, NN, dim)),
                (MM, MM),
                method=tf.image.ResizeMethod.BICUBIC,
            )
            patch_pos_embed = tf.reshape(patch_pos_embed, (1, -1, dim))
            return tf.concat((class_pos_embed, patch_pos_embed), axis=1)

        return tf.cond(tf.equal(M,N),lambda: self.pe, interpolate)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, *args, num_heads, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        if isinstance(input_shape,list):
            if all(isinstance(x,tf.TensorShape) for x in input_shape):
                if len(input_shape) == 1:
                    query_hidden_size = key_hidden_size = value_hidden_size = input_shape[0][-1]
                elif len(input_shape) == 2:
                    query_hidden_size, key_hidden_size, value_hidden_size = input_shape[0][-1], input_shape[1][-1], input_shape[1][-1]
                elif len(input_shape) == 3:
                    query_hidden_size, key_hidden_size, value_hidden_size = input_shape[0][-1], input_shape[1][-1], input_shape[2][-1]
                else:
                    raise ValueError(f"[ERR]: inputs list {input_shape} must be no more than 3 elements")
            else:
                raise ValueError(f"[ERR]: input must be tf.TensorShape or a list of tf.TensorShape, not {input_shape}")
        elif isinstance(input_shape,tf.TensorShape):
            query_hidden_size = key_hidden_size = value_hidden_size = input_shape[-1]
        else:
            raise ValueError(f"[ERR]: input_shape {input_shape} does not match supported types.")

        num_heads = self.num_heads
        hidden_size = query_hidden_size
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {hidden_size} should be divisible by number of heads = {num_heads}"
            )
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // num_heads
        self.query_dense = tf.keras.layers.Dense(query_hidden_size, name="query")
        self.key_dense = tf.keras.layers.Dense(key_hidden_size, name="key")
        self.value_dense = tf.keras.layers.Dense(value_hidden_size, name="value")
        self.combine_heads = tf.keras.layers.Dense(hidden_size, name="out")

    # pylint: disable=no-self-use
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, value=None, key=None):
        if value is None:
            value = query
        if key is None:
            key = value

        batch_size = tf.shape(query)[0]
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.hidden_size))
        output = self.combine_heads(concat_attention)
        return output, weights

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# pylint: disable=too-many-instance-attributes
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    """Implements a Transformer block."""

    def __init__(self, *args, num_heads, mlp_dim, dropout, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

    def build(self, input_shape):
        
        if isinstance(input_shape,list):
            if all(isinstance(x,tf.TensorShape) for x in input_shape):
                hidden_size = input_shape[0][-1]
            else:
                raise ValueError(f"[ERR]: input_shape list {input_shape} does not match supported type tf.TensorShape.")
        elif isinstance(input_shape, tf.TensorShape):
            hidden_size = input_shape[-1]
        else:
            raise ValueError(f"[ERR]: input_shape {input_shape} does not match supported types.")

        self.att = MultiHeadSelfAttention(
            num_heads=self.num_heads,
            name="MultiHeadDotProductAttention_1",
        )
        self.mlpblock = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.mlp_dim,
                    activation="linear",
                    name=f"{self.name}/Dense_0",
                ),
                tf.keras.layers.Lambda(
                    lambda x: tf.keras.activations.gelu(x, approximate=False)
                )
                if hasattr(tf.keras.activations, "gelu")
                else tf.keras.layers.Lambda(
                    lambda x: tfa.activations.gelu(x, approximate=False)
                ),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Dense(hidden_size, name=f"{self.name}/Dense_1"),
                tf.keras.layers.Dropout(self.dropout),
            ],
            name="MlpBlock_3",
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_0"
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_2"
        )
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

    def call(self, inputs, training):
        if isinstance(inputs,list):
            query, *rest = inputs
        else:
            query = inputs
        x = [self.layernorm1(query)]
        if isinstance(inputs,list):
            x += rest
        x, weights = self.att(*x)
        x = self.dropout_layer(x, training=training)
        x = x + query
        y = self.layernorm2(x)
        y = self.mlpblock(y)
        return x + y, weights

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout": self.dropout,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
