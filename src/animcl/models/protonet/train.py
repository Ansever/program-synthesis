import os
import json
from functools import partial
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from tqdm import tqdm

import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from animcl.models.protonet.engine import Engine, State
from animcl.models.protonet.utils import load_data
import torchnet as tnt


import protonets.utils.model as model_utils
import protonets.utils.log as log_utils


class DataArgs(BaseModel):
    dataset: str = Field(default="omniglot", description="data set name")
    split: str = Field(default="vinyals", description="split name")
    way: int = Field(default=60, description="number of classes per episode")
    shot: int = Field(default=5, description="number of support examples per class")
    query: int = Field(default=5, description="number of query examples per class")
    test_way: int = Field(
        default=5,
        description="number of classes per episode in test. 0 means same as data.way",
    )
    test_shot: int = Field(
        default=0,
        description="number of support examples per class in test. 0 means same as data.shot",
    )
    test_query: int = Field(
        default=15,
        description="number of query examples per class in test. 0 means same as data.query",
    )
    train_episodes: int = Field(
        default=100, description="number of train episodes per epoch"
    )
    test_episodes: int = Field(
        default=100, description="number of test episodes per epoch"
    )
    trainval: bool = Field(default=False, description="run in train+validation mode")
    sequential: bool = Field(
        default=False, description="use sequential sampler instead of episodic"
    )
    cuda: bool = Field(default=False, description="run in CUDA mode")


class ModelArgs(BaseModel):
    model_name: str = Field(default="protonet_conv", description="model name")
    x_dim: tuple[int, int, int] = Field(
        default=(1, 28, 28), description="dimensionality of input images"
    )
    hid_dim: int = Field(default=64, description="dimensionality of hidden layers")
    z_dim: int = Field(default=64, description="dimensionality of input images")


class TrainArgs(BaseModel):
    epochs: int = Field(default=10000, description="number of epochs to train")
    optim_method: str = Field(default="Adam", description="optimization method")
    learning_rate: float = Field(default=0.001, description="learning rate")
    decay_every: int = Field(
        default=20,
        description="number of epochs after which to decay the learning rate",
    )
    weight_decay: float = Field(default=0.0, description="weight decay")
    patience: int = Field(
        default=200,
        description="number of epochs to wait before validation improvement",
    )


class LogSettings(BaseModel):
    fields: str = Field(
        default="loss,acc", description="fields to monitor during training"
    )
    exp_dir: str = Field(
        default="results", description="directory where experiments should be saved"
    )


class TrainArgs(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)

    data: DataArgs = Field(default_factory=DataArgs)
    model: ModelArgs = Field(default_factory=ModelArgs)
    train: TrainArgs = Field(default_factory=TrainArgs)
    log: LogSettings = Field(default_factory=LogSettings)


def main(args: TrainArgs):
    if not os.path.isdir(args.log.exp_dir):
        os.makedirs(args.log.exp_dir)

    # save opts
    with open(os.path.join(args.log.exp_dir, "opt.json"), "w") as f:
        json.dump(args, f)
        f.write("\n")

    trace_file = os.path.join(args.log.exp_dir, "trace.txt")

    # Postprocess arguments
    args.model.x_dim = list(map(int, args.model.x_dim.split(",")))
    args.log.fields = args.log.fields.split(",")

    torch.manual_seed(1234)
    if args.data.cuda:
        torch.cuda.manual_seed(1234)

    if args.data.trainval:
        data = load_data(args, ["trainval"])
        train_loader = data["trainval"]
        val_loader = None
    else:
        data = load_data(args, ["train", "val"])
        train_loader = data["train"]
        val_loader = data["val"]

    model = model_utils.load(args)

    if args.data.cuda:
        model.cuda()

    engine = Engine()

    meters = {
        "train": {field: tnt.meter.AverageValueMeter() for field in args.log.fields}
    }

    if val_loader is not None:
        meters["val"] = {
            field: tnt.meter.AverageValueMeter() for field in args.log.fields
        }

    def on_start(state: State):
        if os.path.isfile(trace_file):
            os.remove(trace_file)
        state["scheduler"] = lr_scheduler.StepLR(
            state["optimizer"], args.train.decay_every, gamma=0.5
        )

    engine.hooks["on_start"] = on_start

    def on_start_epoch(state: State):
        for split, split_meters in meters.items():
            for field, meter in split_meters.items():
                meter.reset()
        state["scheduler"].step()

    engine.hooks["on_start_epoch"] = on_start_epoch

    def on_update(state: State):
        for field, meter in meters["train"].items():
            meter.add(state["output"][field])

    engine.hooks["on_update"] = on_update

    def on_end_epoch(hook_state, state: State):
        if val_loader is not None:
            if "best_loss" not in hook_state:
                hook_state["best_loss"] = np.inf
            if "wait" not in hook_state:
                hook_state["wait"] = 0

        if val_loader is not None:
            model_utils.evaluate(
                state["model"],
                val_loader,
                meters["val"],
                desc="Epoch {:d} valid".format(state["epoch"]),
            )

        meter_vals = log_utils.extract_meter_values(meters)
        print(
            "Epoch {:02d}: {:s}".format(
                state["epoch"], log_utils.render_meter_values(meter_vals)
            )
        )
        meter_vals["epoch"] = state["epoch"]
        with open(trace_file, "a") as f:
            json.dump(meter_vals, f)
            f.write("\n")

        if val_loader is not None:
            if meter_vals["val"]["loss"] < hook_state["best_loss"]:
                hook_state["best_loss"] = meter_vals["val"]["loss"]
                print(
                    "==> best model (loss = {:0.6f}), saving model...".format(
                        hook_state["best_loss"]
                    )
                )

                state["model"].cpu()
                torch.save(
                    state["model"], os.path.join(args.log.exp_dir, "best_model.pt")
                )
                if args.data.cuda:
                    state["model"].cuda()

                hook_state["wait"] = 0
            else:
                hook_state["wait"] += 1

                if hook_state["wait"] > args.train.patience:
                    print("==> patience {:d} exceeded".format(args.train.patience))
                    state["stop"] = True
        else:
            state["model"].cpu()
            torch.save(state["model"], os.path.join(args.log.exp_dir, "best_model.pt"))
            if args.data.cuda:
                state["model"].cuda()

    engine.hooks["on_end_epoch"] = partial(on_end_epoch, {})

    engine.train(
        model=model,
        loader=train_loader,
        optim_method=getattr(optim, args.train.optim_method),
        optim_config={
            "lr": args.train.learning_rate,
            "weight_decay": args.train.weight_decay,
        },
        max_epoch=args.train.epochs,
    )


if __name__ == "__main__":
    args = TrainArgs()
    main(args)
