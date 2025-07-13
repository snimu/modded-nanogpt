import re
import matplotlib.pyplot as plt
from typing import Literal

TARGET_STEPS = 5960          # the run length we care about

# ──────────────────────────────────────────────────────────────
# 1.  Parse → {(input, valemb): [(step, t_sec, val_loss), …]}
# ──────────────────────────────────────────────────────────────
def _parse_results(text: str, *, target_steps: int = TARGET_STEPS):
    """
    Extract only the lines whose denominator == target_steps (default 5960).
    """
    recs = {}
    text = text.replace('\r', '')                              # normalise NLs
    for block in re.split(r'\n## +', text):
        if not block.strip() or block.lstrip().startswith('#'):
            continue
        header, *_ = block.splitlines()
        m = re.match(r'.*?(toks|bytes|mot)-in_(toks|bytes|mot)-valemb', header)
        if not m:
            continue
        key = tuple(m.groups())

        runs = [
            (int(step), int(ms) / 1_000.0, float(vloss))       # ms → sec
            for step, denom, vloss, ms in re.findall(
                r'step:(\d+)/(\d+)\s+val_loss:([0-9.]+)\s+train_time:(\d+)ms',
                block
            )
            if int(denom) == target_steps                      # keep only 5960
        ]
        if runs:                                               # skip empty sets
            recs[key] = runs
    return recs


# ──────────────────────────────────────────────────────────────
# 2.  Plotting helper
# ──────────────────────────────────────────────────────────────
def plot_val_loss(
    results_text: str,
    *,
    input=None,                       # 'mot' | 'bytes' | 'toks' | list | None
    valemb=None,                      # same as input
    plot_over: Literal["step", "time"] = "step",
):
    """Plot validation-loss curves, colour = input, linestyle = Value-Embedding."""
    pretty = dict(mot="MoT", bytes="Bytes", toks="Tokens")

    # colour-blind–safe palette (Paul Tol, “bright”)
    colour = {'mot': '#0072B2', 'bytes': '#D55E00', 'toks': '#009E73'}

    style  = {'mot': '--', 'bytes': ':', 'toks': '-'}

    recs = _parse_results(results_text)

    mkset = lambda x: None if x is None else {x} if isinstance(x, str) else set(x)
    keep_inp, keep_emb = map(mkset, (input, valemb))

    keys = [
        k for k in recs
        if (keep_inp is None or k[0] in keep_inp)
        and (keep_emb is None or k[1] in keep_emb)
    ]
    if not keys:
        raise ValueError("No runs match the given filters (or no 5960-step data).")

    plt.figure(figsize=(10, 6))
    for inp, emb in sorted(keys):
        steps, secs, losses = zip(*recs[(inp, emb)])
        x = steps if plot_over == "step" else secs
        plt.plot(
            x, losses,
            label=f"{pretty[inp]} Input, {pretty[emb]} Value Embeddings",
            color=colour[inp],
            linestyle=style[emb],
            linewidth=2,
        )

    plt.title(f"Validation Loss – {TARGET_STEPS}-step Runs")
    plt.xlabel("Step" if plot_over == "step" else "Training Time (s)")
    plt.ylabel("Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()



def main():
    with open("results.md", "r") as f:
        results_text = f.read()
    plot_val_loss(results_text, plot_over='time', input=['mot', 'toks'], valemb=['mot', 'toks'])


if __name__ == "__main__":
    main()
