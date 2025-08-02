
from typing import Literal

import matplotlib.pyplot as plt
import math
import numpy as np


def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


def get_window_size_blocks(
        step: int, *,
        max_window_size: int = 3456,
        step_size: int = 128,
        schedule: Literal["cubic", "sqrt"] = "cubic",
):
    x = step / 5960 # progress in training
    assert 0 <= x <= 1
    # Linearly increase the block-wise sliding window size over training 128 -> 1792
    # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
    if schedule == "cubic":
        factor = 4 * x ** 3 - 6 * x ** 2 + 3 * x # cubic schedule by @jadenj3o
    elif schedule == "sqrt":
        factor = math.sqrt(x * (2 - x))
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    return next_multiple_of_n(max_window_size * factor, n=step_size)


def get_lr(step: int, *, num_iterations: int = 5960, cooldown_frac: float = 0.7):
    x = step / num_iterations # progress in training
    assert 0 <= x < 1
    if x < 1 - cooldown_frac:
        return 1.0
    else:
        return (1 - x) / cooldown_frac

def plot_hparams(
        schedule: Literal["cubic", "sqrt"] | list[Literal["cubic", "sqrt"]] = "cubic",
        plot_lr: bool = True
):
    x = list(range(5960))
    max_window_size: int = 3456
    schedule = [schedule] if isinstance(schedule, str) else schedule
    for s in schedule:
        assert s in ["cubic", "sqrt"]
        ws = [get_window_size_blocks(i, schedule=s) / max_window_size for i in x]
        plt.plot(x, ws, label=f"train seq len with {s} schedule")
    if plot_lr:
        lr = [get_lr(i) for i in x]
        plt.plot(x, lr, label="learning rate")
    plt.xlabel("step")
    plt.ylabel(r"% of maximum")
    plt.legend()
    plt.grid()
    plt.show()


def plot_results(
        header_numbers: list[int | str] | dict[int | str, str],
        filename: str,
        x_axis: str = "step",
):
    with open(filename, "r") as f:
        lines = f.readlines()

    if isinstance(header_numbers, dict):
        descriptions = list(header_numbers.values())
        header_numbers = list(header_numbers.keys())
    else:
        descriptions = ["" for _ in header_numbers]

    parsed = {hnum: {"step": [], "time": [], "loss": []} for hnum in header_numbers}
    for hnum in header_numbers:
        extract= False
        for line in lines:
            if line.startswith(f"## {hnum}_"):
                extract = True
                continue
            if extract and line.startswith("##"):
                break
            if extract and line.startswith("step:"):
                parsed[hnum]["loss"].append(float(line.split()[1].split("val_loss:")[-1]))
                parsed[hnum]["step"].append(int(line.split("step:")[1].split("/")[0]))
                parsed[hnum]["time"].append(float(line.split("train_time:")[1].split("ms")[0]) / 1000)
    
    for i, hnum in enumerate(header_numbers):
        description = f": {descriptions[i]}" if descriptions[i] else ""
        plt.plot(parsed[hnum][x_axis], parsed[hnum]["loss"], label=f"{hnum}{description}")
    plt.xlabel("step" if x_axis == "step" else "time (s)")
    plt.ylabel("val_loss")
    plt.legend()
    plt.grid()
    plt.show()


def plot_byte_stats(header_numbers: list[int | str], filename: str, x_axis: str = "step"):
    with open(filename, "r") as f:
        lines = f.readlines()
    
    parsed = {hnum: {"step": [], "bytes_total": [], "bytes_pulled": [], "bytes_blocked": []} for hnum in header_numbers}
    for hnum in header_numbers:
        extract= False
        for line in lines:
            if line.startswith(f"## {hnum}_"):
                extract = True
                continue
            if extract and line.startswith("##"):
                break
            if extract and line.startswith("step:"):
                parsed[hnum]["bytes_total"].append(int(line.split("total_bytes:")[1].split(" ")[0].replace("_", "")))
                parsed[hnum]["bytes_pulled"].append(int(line.split("total_pulled:")[1].split(" ")[0].replace("_", "")))
                parsed[hnum]["bytes_blocked"].append(int(line.split("total_blocked:")[1].split(" ")[0].replace("_", "")))
                parsed[hnum]["step"].append(int(line.split("step:")[1].split("/")[0]))

    for hnum in header_numbers:
        plt.plot(parsed[hnum][x_axis], np.array(parsed[hnum]["bytes_total"]) / max(parsed[hnum]["bytes_total"]), label=f"{hnum}: total")
        plt.plot(parsed[hnum][x_axis], np.array(parsed[hnum]["bytes_pulled"]) / max(parsed[hnum]["bytes_total"]), label=f"{hnum}: pulled")
        plt.plot(parsed[hnum][x_axis], np.array(parsed[hnum]["bytes_blocked"]) / max(parsed[hnum]["bytes_total"]), label=f"{hnum}: blocked")
    plt.xlabel("step" if x_axis == "step" else "time (s)")
    plt.legend()
    plt.title("Byte stats: total, pulled, blocked")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # plot_hparams(schedule=["cubic", "sqrt"], plot_lr=False)
    plot_results(
        {
            # 0: "Baseline",
            1: "Baseline",
            # 7: "MoT",
            # 71: "MoT-sum",
            # 72: "MoT, hparams",
            # 73: "MoT-sum, norm-then-sum",
            # 74: "MoT-sum, norm-then-sum with lambdas",
            # 75: "MoT, hparams, token_dim=896",
            # "01": "Baseline, lr-schedule",
            # 76: "MoT, hparams, token_dim=896, lr-schedule",
            # "02": "Baseline, shuffled data",
            # 77: "MoT, hparams, token_dim=896, shuffled data",
            # "03": "Baseline, seq-len schedule",
            # 78: "MoT, hparams, token_dim=896, seq-len schedule",
            711: "MoT-concat",
            712: "MoT-concat, Add 758 to MLP hidden dim",
        },
        filename="results.md",
        x_axis="time",
    )
    # plot_byte_stats([79], "results.md", x_axis="step")