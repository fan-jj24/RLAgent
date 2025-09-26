from typing import Literal, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax


def make_attn_mask(input_mask, mask_ar):
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)

class PositionalEncoding(nn.Module):
    d_model: int
    max_len: int
    kind: Literal["sin", "learned"] = "sin"

    @nn.compact
    def __call__(self, x) :
        # x: (B, L, C)
        B, L, C = x.shape
        assert C == self.d_model
        if self.kind == "learned":
            pe = self.param("pos_embed",
                            nn.initializers.normal(0.02),
                            (self.max_len, self.d_model))[:L]
        else:
            position = jnp.arange(L)[:, None]  # (L,1)
            div_term = jnp.exp(
                jnp.arange(0, self.d_model, 2) * (-jnp.log(10000.0) / self.d_model)
            )
            pe = jnp.zeros((L, self.d_model))
            pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
            pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        return x + pe[None, :, :]
    

class StateToTemporalGaussian(nn.Module):
    horizon: int = 3
    out_dim: int = 32
    model_dim: int = 128
    n_heads: int = 4
    pos_kind: Literal["sin", "learned"] = "learned"
    dropout_rate: float = 0.0
    
    def setup(self):
        self.pos_enc = PositionalEncoding(d_model=self.model_dim, max_len=1000, kind=self.pos_kind)

    @nn.compact
    def __call__(self, state):
        embedded, inputmask, armask = state
        attn_mask = make_attn_mask(inputmask, armask)
        # print(embedded.shape)#(B, 816, 2048)
        # print(inputmask.shape)#(B, 816)
        # print(armask.shape)#(816,)
        # print(attn_mask.shape)#(B, 816, 816)
        attn_mask = attn_mask[:, None, :, :]  # (B, 1, 816, 816)
       
        h = nn.Dense(features=self.model_dim, name="proj")(embedded)
        h = nn.gelu(h)
        h = nn.LayerNorm(name="ln_proj")(h)

        h = self.pos_enc(h)

        h_attn = nn.SelfAttention(
            num_heads=self.n_heads,
            qkv_features=self.model_dim,
            out_features=self.model_dim,
            dropout_rate=self.dropout_rate,
            deterministic=False,
            name=f"attn"
        )(h, mask=attn_mask)
        h = h + h_attn
        h = nn.LayerNorm(name=f"ln_attn")(h)

        h_ffn = nn.Dense(features=4 * self.model_dim, name=f"dense_ffn0")(h)
        h_ffn = nn.gelu(h_ffn)
        h_ffn = nn.Dense(features=self.model_dim, name=f"dense_ffn1")(h_ffn)
        h = h + h_ffn
        h = nn.LayerNorm(name=f"ln_ffn")(h) #(B, 816, 128)

        h = jnp.mean(h.reshape(h.shape[0], self.horizon, -1, h.shape[-1]), axis=2) #(B, 3, 128)

        mu = nn.Dense(self.out_dim, name="mu_head",kernel_init=nn.initializers.variance_scaling(0.001, mode="fan_in", distribution="normal"))(h) #(B, 3, 32)
        mu = 3.0 * jax.nn.tanh(mu)

        std = nn.Dense(self.out_dim, name="std_head")(h) #(B, 3, 32)
        low, high = jnp.log(0.1), jnp.log(1.5)
        std = low + (high - low) * jax.nn.sigmoid(std)
        std = jnp.exp(std)

        base_dist = distrax.Normal(loc=mu, scale=std)
        return distrax.Independent(base_dist, reinterpreted_batch_ndims=2)