from typing import Literal, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax

from serl_launcher.networks.lagrange import GeqLagrangeMultiplier

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
    horizon: int = 50
    out_dim: int = 32
    model_dim: int = 128
    n_heads: int = 4
    pos_kind: Literal["sin", "learned"] = "learned"
    dropout_rate: float = 0.0
    
    def setup(self):
        # self.std_module = GeqLagrangeMultiplier(init_value=1.1, constraint_shape=(self.horizon, self.out_dim))
        self.pos_enc = PositionalEncoding(d_model=2048, max_len=1000, kind=self.pos_kind)

    @nn.compact
    def __call__(self, state):
        # state: (embedded, inputmask, armask)
        embedded, inputmask, armask = state
        attn_mask = make_attn_mask(inputmask, armask)
        # print(h.shape)#(B, L, 2048)
        # print(inputmask.shape)#(B, L)
        # print(armask.shape)#(L,)
        # print(attn_mask.shape)#(B, L, L)
        attn_mask = attn_mask[:, None, :, :]  # (B, 1, L, L)
        B, L, D = embedded.shape
        h = self.pos_enc(embedded)
        
        h_attn = nn.SelfAttention(
            num_heads=self.n_heads,
            qkv_features=D,
            out_features=D,
            dropout_rate=self.dropout_rate,
            deterministic=False,
            name=f"attn"
        )(h, mask=attn_mask)
        h = h + h_attn
        h = nn.LayerNorm(name=f"ln_attn")(h)

        h_ffn = nn.Dense(features=4 * D, name=f"dense_ffn0")(h)
        h_ffn = nn.gelu(h_ffn)
        h_ffn = nn.Dense(features=D, name=f"dense_ffn1")(h_ffn)
        h = h + h_ffn
        h = nn.LayerNorm(name=f"ln_ffn")(h)

        h = nn.Dense(features=self.model_dim, name="proj")(h)
        h = nn.gelu(h)
        h = nn.LayerNorm(name="ln_proj")(h)
        h = h.reshape(B, -1) #(B, L * d_model)
        
        mu = nn.Dense(self.horizon * self.out_dim, name="mu_head")(h) #(B, H * out_dim)
        low, high = -3, 3
        mu = low + (high - low) * jax.nn.sigmoid(mu)
        mu = mu.reshape(B, self.horizon, self.out_dim)
        
        std = nn.Dense(self.horizon * self.out_dim, name="std_head")(h)
        low, high = jnp.log(0.1), jnp.log(3)
        std = low + (high - low) * jax.nn.sigmoid(std)
        std = jnp.exp(std)
        # std = jnp.exp(jnp.clip(std, jnp.log(0.1), jnp.log(1.5)))
        # std = jnp.clip(jnp.exp(std), 0.1, 1.2)
        std = std.reshape(B, self.horizon, self.out_dim)
        '''
        std_param = self.std_module()
        std = jnp.broadcast_to(std_param, (B, self.horizon, self.out_dim))
        '''
        base_dist = distrax.Normal(loc=mu, scale=std)
        return distrax.Independent(base_dist, reinterpreted_batch_ndims=2)