import torch
from typing import Any, Callable, TypedDict
from tqdm import tqdm

from animcl.models.protonet.model import Protonet

type Hook = Callable[[State], Any]


class Hooks(TypedDict):
    on_start: Hook
    on_start_epoch: Hook
    on_sample: Hook
    on_forward: Hook
    on_backward: Hook
    on_end_epoch: Hook
    on_update: Hook
    on_end: Hook


class State(TypedDict):
    model: Protonet
    loader: torch.utils.data.DataLoader
    optim_method: type[torch.optim.Optimizer]
    optim_config: dict[str, Any]
    max_epoch: int
    epoch: int  # epochs done so far
    t: int  # samples seen so far
    batch: int  # samples seen in current epoch
    stop: bool
    optimizer: torch.optim.Optimizer
    sample: Any
    output: Any
    epoch_size: int
    loss: Any


class Engine:
    def __init__(self) -> None:
        def _empty_hook(state: State) -> None:
            pass

        self.hooks: Hooks = {
            "on_start": _empty_hook,
            "on_start_epoch": _empty_hook,
            "on_sample": _empty_hook,
            "on_forward": _empty_hook,
            "on_backward": _empty_hook,
            "on_end_epoch": _empty_hook,
            "on_update": _empty_hook,
            "on_end": _empty_hook,
        }

    def train(
        self,
        model: Protonet,
        loader: torch.utils.data.DataLoader,
        optim_method: type[torch.optim.Optimizer],
        optim_config: dict[str, Any],
        max_epoch: int,
    ):
        state: State = {
            "model": model,
            "loader": loader,
            "optim_method": optim_method,
            "optim_config": optim_config,
            "max_epoch": max_epoch,
            "epoch": 0,  # epochs done so far
            "t": 0,  # samples seen so far
            "batch": 0,  # samples seen in current epoch
            "stop": False,
        }

        state["optimizer"] = state["optim_method"](
            state["model"].parameters(), **state["optim_config"]
        )

        self.hooks["on_start"](state)
        while state["epoch"] < state["max_epoch"] and not state["stop"]:
            state["model"].train()

            self.hooks["on_start_epoch"](state)

            state["epoch_size"] = len(state["loader"])

            for sample in tqdm(
                state["loader"], desc="Epoch {:d} train".format(state["epoch"] + 1)
            ):
                state["sample"] = sample
                self.hooks["on_sample"](state)

                state["optimizer"].zero_grad()
                loss, state["output"] = state["model"].loss(state["sample"])
                self.hooks["on_forward"](state)

                loss.backward()
                self.hooks["on_backward"](state)

                state["optimizer"].step()

                state["t"] += 1
                state["batch"] += 1
                self.hooks["on_update"](state)

            state["epoch"] += 1
            state["batch"] = 0
            self.hooks["on_end_epoch"](state)

        self.hooks["on_end"](state)
