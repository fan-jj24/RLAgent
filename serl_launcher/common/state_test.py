import functools
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, Union, Optional
from flax.core import FrozenDict
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct
import numpy as np
from serl_launcher.common.typing import Params, PRNGKey

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)
default_init = nn.initializers.xavier_uniform

class JaxRLTrainState(struct.PyTreeNode):
    ACTOR_PARAM_KEY = "modules_actor"
    CRITIC_PARAM_KEY = "modules_critic"
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: Params
    target_params: Params
    txs: Any = struct.field(pytree_node=False)
    opt_states: Any
    rng: PRNGKey
    epsilon: float = 0.0

    def target_update(self, tau: float) -> "JaxRLTrainState":
        current_critic = self.params['modules_critic']
        target_critic = self.target_params['modules_critic']
        new_critic = jax.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            current_critic,
            target_critic
        )
        return self.replace(target_params={"modules_critic": new_critic})

    def apply_gradients(self, *, grads: Any, active_network: Optional[str] = None) -> "JaxRLTrainState":
        tx = self.txs[active_network]
        opt_state = self.opt_states[active_network]
        grad = grads[active_network]
        if active_network == self.ACTOR_PARAM_KEY:
            update, new_opt_state = tx.update(grad, opt_state, self.params)
            new_params = optax.apply_updates(self.params, update)
        elif active_network == self.CRITIC_PARAM_KEY:
            update, new_opt_state = tx.update(grad, opt_state, self.params[active_network])
            new_critic_params = optax.apply_updates(self.params[active_network], update)
            new_params = dict(self.params)  
            new_params[active_network] = new_critic_params 
        
        new_opt_states = dict(self.opt_states)
        new_opt_states[active_network] = new_opt_state
        
        return self.replace(
            step=self.step + 1, 
            params=new_params, 
            opt_states=new_opt_states
        )

    def apply_loss_fns(
        self, loss_fns: Any, has_aux: bool = False, active_network: Optional[str] = None
    ) -> Union["JaxRLTrainState", Tuple["JaxRLTrainState", Any]]:
        def grad_fn(key, loss_fn, rng):
            def wrapped_loss_fn(params, rng):
                if key == self.ACTOR_PARAM_KEY:
                    return loss_fn(params, rng)
                elif key == self.CRITIC_PARAM_KEY:
                    return loss_fn(params[key], rng)           
            return jax.grad(wrapped_loss_fn, has_aux=has_aux)(self.params, rng)

        treedef = jax.tree_util.tree_structure(loss_fns)
        new_rng, *rngs = jax.random.split(self.rng, treedef.num_leaves + 1)
        rngs = jax.tree_util.tree_unflatten(treedef, rngs)
        
        grads_and_aux = {}
        for key, loss_fn in loss_fns.items():
            rng = rngs[key] if isinstance(rngs, dict) else rngs 
            raw_tuple = grad_fn(key, loss_fn, rng)
            if key == self.ACTOR_PARAM_KEY:
                grads_and_aux[key] = tuple(
                    (raw_tuple[0],
                    raw_tuple[1])
                )
            elif key == self.CRITIC_PARAM_KEY:
                grads_and_aux[key] = tuple(
                    (raw_tuple[0][self.CRITIC_PARAM_KEY],
                    raw_tuple[1])
                )

        self = self.replace(rng=new_rng)
        if has_aux:
            grads = {k: v[0] for k, v in grads_and_aux.items()}
            aux = {k: v[1] for k, v in grads_and_aux.items()}
            return self.apply_gradients(grads=grads, active_network=active_network), aux
        else:
            return self.apply_gradients(grads=grads_and_aux, active_network=active_network)

    @classmethod
    def create(
        cls, *, apply_fn, params, txs, target_params=None, rng=jax.random.PRNGKey(0), epsilon=0.0
    ):
        if txs:
            opt_states = {
                cls.ACTOR_PARAM_KEY: txs[cls.ACTOR_PARAM_KEY].init(params),
                cls.CRITIC_PARAM_KEY: txs[cls.CRITIC_PARAM_KEY].init(params[cls.CRITIC_PARAM_KEY]),
            }
        else:
            opt_states = None
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            target_params=target_params,
            txs=txs,
            opt_states=opt_states,
            rng=rng,
            epsilon=epsilon,
        )
