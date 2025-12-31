import tensorflow as tf

#causal self attention layer
class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, num_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = tf.keras.layers.Dense(hidden_dim * 3)
        self.proj = tf.keras.layers.Dense(hidden_dim)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (B, T, 3, self.num_heads, self.head_dim))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = tf.matmul(q, k, transpose_b=True)
        att = att * self.scale
        mask = tf.linalg.band_part(tf.ones((T, T)), -1, 0)
        mask = tf.reshape(mask, (1, 1, T, T))
        att = att * mask + (1 - mask) * -1e9
        att = tf.nn.softmax(att, axis=-1)
        att = self.dropout(att, training=training)
        y = tf.matmul(att, v)
        y = tf.transpose(y, (0, 2, 1, 3))
        y = tf.reshape(y, (B, T, self.hidden_dim))
        return self.proj(y)

# transformer block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, dropout)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ffn_dim, activation="relu"),
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.Dropout(dropout)
        ])
  
    def call(self, x, training=False):
        x = x + self.attn(self.ln1(x), training=training)
        x = x + self.ffn(self.ln2(x), training=training)
        return x

# defining and using layers
class TransformerLM(tf.keras.Model):
    def __init__(self, vocab_size, max_len, num_layers=4, num_heads=4, hidden_dim=256, ffn_dim=1024, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout
        self.token_emb = tf.keras.layers.Embedding(vocab_size, hidden_dim)
        self.pos_emb = tf.keras.layers.Embedding(max_len, hidden_dim)
        self.blocks = [TransformerBlock(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.head = tf.keras.layers.Dense(vocab_size)

    def call(self, x, training=False):
        T = tf.shape(x)[1]

        pos = tf.range(T)[tf.newaxis, :]
        h = self.token_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h, training=training)
        h = self.ln_f(h)
        return self.head(h)
