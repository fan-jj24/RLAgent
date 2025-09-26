import functools
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, Union, Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct

class ModuleDict(nn.Module):

    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f"When `name` is not specified, kwargs must contain the arguments for each module. "
                    f"Got kwargs keys {kwargs.keys()} but module keys {self.modules.keys()}"
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out

        return self.modules[name](*args, **kwargs)
    
    def call_method(self, *args, name=None, method_name=None, **kwargs):
        module = self.modules[name]
        method = getattr(module, method_name)
        return method(*args, **kwargs)
        
class JaxRLTrainState(struct.PyTreeNode):

    step: int
    apply_fn:Callable = struct.field(pytree_node=False)
    params: Any
    policy_params: Any
    tx: Any = struct.field(pytree_node=False)
    opt_state: Any
    rng: Any

    def apply_gradients(self, *, grads: Any) -> "JaxRLTrainState":
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1, params=new_params, opt_state=new_opt_state
        )

    @classmethod
    def create(
        cls, *, apply_fn, params, tx, policy_params=None, rng=jax.random.PRNGKey(0)
    ):
        policy_params = {**policy_params, **policy_params["PaliGemma"]}
        del policy_params["PaliGemma"]

        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            policy_params={"modules_policy":policy_params},
            tx=tx,
            opt_state=tx.init(params),
            rng=rng,
        )
