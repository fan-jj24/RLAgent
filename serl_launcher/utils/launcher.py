# !/usr/bin/env python3
from agentlace.trainer import TrainerConfig
from serl_launcher.common.wandb import WandBLogger
    
def make_trainer_config(port_number: int = 5588, broadcast_port: int = 5589):
    return TrainerConfig(
        port_number=port_number,
        broadcast_port=broadcast_port,
        request_types=["register", "get_actions", "record",],
    )


def make_wandb_logger(
    project: str = "hil-serl-for-your-policy",
    description: str = "serl_launcher",
    tag: list[str] = ("serl_launcher",),
    debug: bool = False,
):
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": project,
            "exp_descriptor": description,
            "tag": tag,
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant={},
        debug=debug,
    )
    return wandb_logger
