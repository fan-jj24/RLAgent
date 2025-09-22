import einops
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from serl_launcher.networks import model as _model
from serl_launcher.networks import gemma as _gemma
from serl_launcher.networks import siglip as _siglip
from serl_launcher.common import array_typing as at
from typing_extensions import override

def pad_to_dim(x, target_dim: int, axis: int = -1):
    if isinstance(x, jnp.ndarray):
        lib = jnp
        current_dim = x.shape[axis]
        pad_needed = max(0, target_dim - current_dim)
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, pad_needed)
        return lib.pad(x, pad_width) if pad_needed > 0 else x
    else:
        lib = np
        current_dim = x.shape[axis]
        if current_dim < target_dim:
            pad_width = [(0, 0)] * len(x.shape)
            pad_width[axis] = (0, target_dim - current_dim)
            return lib.pad(x, pad_width)
        return x
    
def input_obs(observation):
    if not isinstance(observation, _model.Observation):
        obs_dict = jax.tree.map(lambda x: x, observation)
        obs_dict = flax.core.unfreeze(obs_dict) 
        obs_dict["image_mask"] = {k: jnp.asarray(v, dtype=bool).reshape(-1) for k, v in obs_dict["image_mask"].items()}
        obs_dict["state"] = jnp.asarray(pad_to_dim(obs_dict["state"], 32), dtype=jnp.float32)
        observation = _model.Observation.from_dict(obs_dict)
    return observation

def make_attn_mask(input_mask, mask_ar):
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float,
                  max_period: float) -> at.Float[at.Array, "b {embedding_dim}"]:
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period)**fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0Config:
    dtype: str = "bfloat16"
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48
    def __init__(self, paligemma_variant: _gemma.Variant = "gemma_2b", action_expert_variant: _gemma.Variant = "gemma_300m"):
        self.paligemma_variant = paligemma_variant
        self.action_expert_variant = action_expert_variant
        
    
class Pi0(nn.Module):
    config: Pi0Config = Pi0Config()

    def setup(self):
        self.action_dim, self.action_horizon, self.max_token_len = self.config.action_dim, self.config.action_horizon, self.config.max_token_len
        paligemma_config = _gemma.get_config(self.config.paligemma_variant)
        action_expert_config = _gemma.get_config(self.config.action_expert_variant)

        self.llm = _gemma.Module(configs=[paligemma_config, action_expert_config], embed_dtype=self.config.dtype)
        self.img = _siglip.Module(
            num_classes=paligemma_config.width,
            variant="So400m/14",
            pool_type="none",
            scan=True,
            dtype_mm=self.config.dtype,
        )

        self.state_proj = nn.Dense(features=action_expert_config.width)
        self.action_in_proj = nn.Dense(features=action_expert_config.width)
        self.action_time_mlp_in = nn.Dense(features=action_expert_config.width)
        self.action_time_mlp_out = nn.Dense(features=action_expert_config.width)
        self.action_out_proj = nn.Dense(features=self.config.action_dim)

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []

        for name in obs.images:
            image_tokens, _ = self.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            ar_mask += [False] * image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.llm.embed(obs.tokenized_prompt)
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask
    
    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []

        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        ar_mask += [True]

        time_emb = posemb_sincos(timestep, self.action_in_proj.features, min_period=4e-3, max_period=4.0)
        action_tokens = self.action_in_proj(noisy_actions)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nn.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)

        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask
    

    def input_obs(self, observation):
        if not isinstance(observation, _model.Observation):
            obs_dict = jax.tree.map(lambda x: x, observation)
            obs_dict = flax.core.unfreeze(obs_dict) 
            obs_dict["image_mask"] = {k: jnp.asarray(v, dtype=bool).reshape(-1) for k, v in obs_dict["image_mask"].items()}
            obs_dict["state"] = jnp.asarray(pad_to_dim(obs_dict["state"], 32), dtype=jnp.float32)
            observation = _model.Observation.from_dict(obs_dict)
        return observation


    def __call__(self, sample_rng, observation, num_steps=10, train=False):
        """
        full denoising diffusion sampling loop: observation -> actions
        """
        sample_rng, rng = jax.random.split(sample_rng)
        observation = self.input_obs(observation)
        observation = _model.preprocess_observation(rng, observation, train=train)

        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(sample_rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.llm(
            embedded=[prefix_tokens, None],
            positions=positions,
            mask=prefix_attn_mask,
        )
        def step(carry, _):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, jnp.broadcast_to(time, batch_size))
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.llm(
                embedded=[None, suffix_tokens],
                positions=positions,
                mask=full_attn_mask,
                kv_cache=kv_cache
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            return (x_t + dt * v_t, time + dt), None

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        # x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        (x_0, _), _ = lax.scan(step, (noise, 1.0), length=num_steps)
        return x_0

    def sample_actions(self, noise, embedding, observation):

        """
        rl: embedding + noise + observation -> actions
        embedding: shared embedding with the generator
        noise: noise generated by the generator
        observation: current observation, used for the 200m action expert
        """
        
        dt = -1.0 / 10
        batch_size = observation.state.shape[0]

        prefix_tokens, prefix_mask, prefix_ar_mask = embedding
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.llm(
            embedded=[prefix_tokens, None],
            positions=positions,
            mask=prefix_attn_mask,
        )
        def step(carry, _):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, jnp.broadcast_to(time, batch_size))
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.llm(
                embedded=[None, suffix_tokens],
                positions=positions,
                mask=full_attn_mask,
                kv_cache=kv_cache
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            return (x_t + dt * v_t, time + dt), None

        (x_0, _), _ = lax.scan(step, (noise, 1.0), length=10)
        return x_0
    
    @override
    def compute_loss(self,
                     rng: at.KeyArrayLike,
                     observation: _model.Observation,
                     actions: _model.Actions,
                     step,
                     *,
                     train: bool = False,
                     ema_model: _model.BaseModel | None = None):
        preprocess_rng, FM_rng, CT_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        CT_WARMUP_STEPS = 5000
        ct_ratio = jnp.minimum(0.4, jnp.maximum(0.1 , step / CT_WARMUP_STEPS))
        batch_size = actions.shape[0]
        num_ct_samples = jnp.floor(ct_ratio * batch_size).astype(jnp.int32)
        num_fm_samples = batch_size - num_ct_samples
        num_ct_samples = 16
        num_fm_samples = 48
        """
        Initially, the plan was to gradually increase the number of CT samples based on the step.
        However, this approach requires using a MASK, which increases computational cost.
        As a result, a fixed number of CT samples is used for now.
        """
        if ema_model is None:
            num_ct_samples = 0
            num_fm_samples = batch_size
        
        actions_ct = jax.lax.dynamic_slice_in_dim(actions, 0, num_ct_samples, axis=0)
        actions_fm = jax.lax.dynamic_slice_in_dim(actions, num_ct_samples, num_fm_samples, axis=0)
        observation_ct = jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(x, 0, num_ct_samples, axis=0),
            observation
        )
        observation_fm = jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(x, num_ct_samples, num_fm_samples, axis=0),
            observation
        )

        FM_loss = self.FM_loss(FM_rng, observation_fm, actions_fm)
        if ema_model is None:
            return jnp.mean(FM_loss)
        else:
            CT_loss = self.CT_loss(CT_rng, observation_ct, actions_ct, ema_model=ema_model)
            return jnp.mean(FM_loss) + jnp.mean(CT_loss)


    def CT_loss(self, CT_rng, observation, actions, ema_model = None):
        if ema_model is None:
            return jnp.zeros(actions.shape[:-2])

        noise_rng, time_rng, time1_rng = jax.random.split(CT_rng, 3)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)

        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm([prefix_tokens, suffix_tokens],
                                                         mask=attn_mask,
                                                         positions=positions)
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        x_t_pred = x_t + (0 - time_expanded) * v_t

        alpha = jax.random.uniform(time1_rng, batch_shape) 
        time1 = alpha * time
        time_expanded1 = time1[..., None, None]
        x_t1 = time_expanded1 * noise + (1 - time_expanded1) * actions
        prefix_tokens1, prefix_mask1, prefix_ar_mask1 = ema_model.embed_prefix(observation)
        suffix_tokens1, suffix_mask1, suffix_ar_mask1 = ema_model.embed_suffix(observation, x_t1, time1)
        input_mask1 = jnp.concatenate([prefix_mask1, suffix_mask1], axis=1)
        ar_mask1 = jnp.concatenate([prefix_ar_mask1, suffix_ar_mask1], axis=0)
        attn_mask1 = make_attn_mask(input_mask1, ar_mask1)
        positions1 = jnp.cumsum(input_mask1, axis=1) - 1
        (prefix_out1, suffix_out1), _ = ema_model.PaliGemma.llm([prefix_tokens1, suffix_tokens1],
                                                         mask=attn_mask1,
                                                         positions=positions1)
        v_t1 = ema_model.action_out_proj(suffix_out1[:, -self.action_horizon:])
        x_t1_pred = x_t1 + (0 - time_expanded1) * v_t1
        x_t1_pred = jax.lax.stop_gradient(x_t1_pred)

        return jnp.mean(jnp.square(x_t1_pred - x_t_pred), axis=-1)


    def FM_loss(self, FM_rng, observation, actions,):
        noise_rng, time_rng = jax.random.split(FM_rng, 2)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm([prefix_tokens, suffix_tokens],
                                                         mask=attn_mask,
                                                         positions=positions)
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)