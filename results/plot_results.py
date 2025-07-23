import matplotlib.pyplot as plt
import math


def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


def get_window_size_blocks(step: int, *, max_window_size: int = 3456, step_size: int = 128):
    x = step / 5960 # progress in training
    assert 0 <= x <= 1
    # Linearly increase the block-wise sliding window size over training 128 -> 1792
    # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
    # factor = 4 * x ** 3 - 6 * x ** 2 + 3 * x # cubic schedule by @jadenj3o
    # factor = 5 * x ** 3 - 6 * x ** 2 + 3 * x # cubic schedule by @jadenj3o
    factor = math.sqrt(x * (2 - x))
    return next_multiple_of_n(max_window_size * factor, n=step_size)


def get_lr(step: int, *, num_iterations: int = 5960, cooldown_frac: float = 0.7):
    x = step / num_iterations # progress in training
    assert 0 <= x < 1
    if x < 1 - cooldown_frac:
        return 1.0
    else:
        return (1 - x) / cooldown_frac

def plot_hparams():
    x = list(range(5960))
    max_window_size: int = 3456
    ws = [get_window_size_blocks(i) / max_window_size for i in x]
    plt.plot(x, ws, label="window size")
    lr = [get_lr(i) for i in x]
    plt.plot(x, lr, label="learning rate")
    plt.legend()
    plt.grid()
    plt.show()


def plot_results(header_numbers: list[int | str], filename: str, x_axis: str = "step"):
    with open(filename, "r") as f:
        lines = f.readlines()
    
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
    
    for hnum in header_numbers:
        plt.plot(parsed[hnum][x_axis], parsed[hnum]["loss"], label=f"{hnum}")
    plt.xlabel("step" if x_axis == "step" else "time (s)")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # plot_hparams()
    plot_results([0, "03", 78], "results.md", x_axis="step")