from typing import Any
import distrax
import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from serl_launcher.common.common import JaxRLTrainState, ModuleDict
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.generator import StateToTemporalGaussian
from serl_launcher.networks.model import restore_params, preprocess_observation
from serl_launcher.networks.pi0 import Pi0, input_obs

class Agent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    config: dict = struct.field(pytree_node=False)
    
    def process_obs(self, observations):
        return preprocess_observation(None, input_obs(observations), train=False)

    def embed_prefix(self, observations):
        return self.state.apply_fn(
            {"params":self.state.policy_params},
            observations,
            name="policy",
            method="call_method",
            method_name="embed_prefix",
        )

    def sample_actions(self, noise, embedding, observation):
        action =  self.state.apply_fn(
            {"params":self.state.policy_params},
            noise,
            embedding,
            observation,
            name="policy",
            method="call_method",
            method_name="sample_actions",
        )
        return jax.tree.map(lambda x: x[..., :14], action)[0]
    
    def forward_generator(self,embeddings,*,grad_params=None):
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            embeddings,
            name="generator",
        )

    def forward_temperature(self, *, grad_params=None):
        return self.state.apply_fn({"params": grad_params or self.state.params}, name="temperature")

    def temperature_lagrange_penalty(self, kl_div, *, grad_params=None):
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            lhs=kl_div,
            rhs=self.config["target_kl_div"],
            name="temperature",
        )

    def loss_fn(self, params, batch):
        temperature_no_grad = jax.lax.stop_gradient(self.forward_temperature())
    
        noise_distributions = self.forward_generator(jax.lax.stop_gradient(self.embed_prefix(self.process_obs(batch["observations"]))), grad_params=params)
        
        mu = noise_distributions.distribution.loc
        sigma = noise_distributions.distribution.scale
        kl_div = jnp.mean(0.5 * (mu**2 + sigma**2 - 1 - 2 * jnp.log(sigma + 1e-6)))
        
        log_probs = noise_distributions.log_prob(batch["noises"])
        ratio = jnp.exp((log_probs - batch["log_probs"]))
        generator_loss = jnp.mean(jnp.minimum(ratio * batch["advantages"], jnp.clip(ratio, 1 - self.config["low_bound"], 1 + self.config["high_bound"]) * batch["advantages"]))
        # generator_loss = jnp.mean(jnp.exp(log_probs) * batch["advantages"])
        '''
        entropy = -jnp.mean(log_probs) / (50 * 32)
        temperature_loss = self.temperature_lagrange_penalty(jax.lax.stop_gradient(kl_div), grad_params=params)'''
        total_loss = ( - generator_loss  + kl_div * self.config["kl_div_coef"]) * temperature_no_grad

        info = {
            "generator_loss": generator_loss,
            # "temperature": temperature_no_grad,
            # "entropy": entropy,
            "kl_div": kl_div,
            "ratio_min": jnp.min(ratio),
            "ratio_max": jnp.max(ratio),
            "dis_mean_min": jnp.min(mu),
            "dis_mean_max": jnp.max(mu),
            "dis_std_min": jnp.min(sigma),
            "dis_std_max": jnp.max(sigma),
        }

        return total_loss, info

    @jax.jit
    def update(self, batch,):
        new_rng, rng = jax.random.split(self.state.rng)
        grads, info = jax.grad(self.loss_fn, has_aux=True)(self.state.params, batch)
        new_state = self.state.apply_gradients(grads=grads).replace(rng=new_rng)
        return self.replace(state=new_state), info

    def interpolate_noise(self, noise):
        positions = jnp.array([0, 24, 49], dtype=jnp.float32)
        x_new = jnp.arange(50, dtype=jnp.float32)
        def interp_along_feature(col):
            return jnp.interp(x_new, positions, col)
        result = jax.vmap(interp_along_feature, in_axes=1, out_axes=1)(noise[0])
        return result[None, ...]
    
    @jax.jit
    def get_actions(self, observation, seed):
        observation = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], observation)
        observation = self.process_obs(observation)
        embedding = self.embed_prefix(observation)
        noise_distribution = self.forward_generator(embedding)
        noise, log_prob = noise_distribution.sample_and_log_prob(seed=seed)
        action = self.sample_actions(self.interpolate_noise(noise), embedding, observation)
        return action, noise, log_prob

    @classmethod
    def create(
        cls,
        rng,
        generator_def,
        temperature_def,
        pretrained_policy_path,
        low_bound,
        high_bound,
        entropy_coef,
        kl_div_coef,
        target_kl_div,    
        temperature_coef,    
    ):
        create_rng, init_rng = jax.random.split(rng)
        policy_def = Pi0()
        rand1 = jax.random.normal(init_rng, (1, 816, 2048))
        rand2 = jax.random.normal(init_rng, (1, 816))
        rand3 = jax.random.normal(init_rng, (816,))
        
        init_embedding = (rand1, rand2, rand3)
        state = JaxRLTrainState.create(
            apply_fn=ModuleDict({"generator": generator_def, "temperature": temperature_def, "policy": policy_def}).apply,
            params=ModuleDict({"generator": generator_def, "temperature": temperature_def,}).init(init_rng,generator=[init_embedding],temperature=[],)["params"],
            tx=make_optimizer(learning_rate=1e-6),
            policy_params=restore_params(pretrained_policy_path, dtype=jnp.bfloat16),
            rng=create_rng,
        )

        return cls(
            state=state,
            config=dict(
                low_bound=low_bound,
                high_bound=high_bound,
                entropy_coef=entropy_coef,
                kl_div_coef=kl_div_coef,
                target_kl_div=target_kl_div,
                temperature_coef=temperature_coef,
            ),
        )

    @classmethod
    def create_pixels(
        cls,
        rng,
        pretrained_policy_path,
        low_bound,
        high_bound,
        entropy_coef,
        kl_div_coef,
        target_kl_div,
        temperature_coef,
    ):
        generator_def = StateToTemporalGaussian()
        temperature_def = GeqLagrangeMultiplier(constraint_shape=(),)
        agent = cls.create(
            rng,
            generator_def=generator_def,
            temperature_def=temperature_def,
            pretrained_policy_path=pretrained_policy_path,
            low_bound=low_bound,
            high_bound=high_bound,
            entropy_coef=entropy_coef,
            kl_div_coef=kl_div_coef,
            target_kl_div=target_kl_div,
            temperature_coef=temperature_coef,
        )

        return agent
