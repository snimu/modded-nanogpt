# Analysis

Analyzing the results as they come in

## 0_train_gpt_medium

The original script.

## 1_toks-in_toks-valemb

The original script but with my dataloader, to see the performance difference it makes.

Performance over time:

![0, 1: over time](images/0_1_time.png)

Performance per step:

![0, 1: over step](images/0_1_step.png)

- There is no difference in per-step performance
  - Good sign for correctness of dataloader
- There is a very slight difference in timing
  - The dataloader isn't as slow as expected
  - But as expected, it does make a bit of a difference

## 7_mot-in_toks-valemb

I'll compare it to 1 as the baseline, so that the dataloader isn't a factor. Let's first look at the per-step and per-time performance of both over the full run.

Per step:

![1, 7: step, full](images/1_7_step_full.png)

- The two are pretty similar
- There is a weird hump around step 3000-4000 in the MoT curve
- The MoT is slightly worse per step

Over time:

![1, 7: time, full](images/1_7_time_full.png)

- Clearly, the MoT is way slower than the baseline
  - This is of course due to the additional re-shaping and the linear layer

Let's look more closely at the first and second half of the per-step plot. The first part:

![1, 7: step, 100-1500](images/1_7_step_100-1500.png)

- The two are neck-on-neck
- The baseline is slightly better most of the way
- Around step 1000, the MoT is slightly better
- Then it gets worse again, and fast

For the second part:

![1, 7: step, 1500-6000](images/1_7_step_1500-6000.png)

- The MoT's loss-curve flattens out, while the baseline's just keeps going
- But at around step 3500, there is a step change in the MoT loss curve and it bends down

This step-change is important, and fixing it would give me a huge boost. It looks like either a hyperparameter thing or some problem with the data.

Let's check out the two hyperparameters that follow a schedule: learning rate and sequence length. And let's normalize them to their maximum size so that they can both fit on the same plot.

![haprams: lr and sequence length](images/hparams_lr-and-seq-len.png)

This doesn't immediately look like the cause of the step change:

- The step change happens at around step 3500
- The learning rate starts decaying before step 2000
- The sequence length is constant for a long time before the step change happens, and only starts increasing after (at maybe step 3900)

So it might be some sort of threshold being reached for the learning rate, but it's still strange, especially because the final loss is very close (baseline: 2.919627, MoT: 2.920585 &rarr; Mot is 1.00032812 larger, or ~0.033%).

## 71_mot-in_toks-valemb

I first want to try a modification of the MoT: instead of concatenating the tokens and bytes and applying a linear layer, I make sure that `byte_dim * bytes_per_token = token_dim = model_dim = 1024` and then sum the tokens and the concatenated bytes of each token. I'll call it MoT-sum. Let's compare this to both 1 and 7. And since I can immediately see that the curves for 7 and 71 are almost exactly the same, I'll just start with the zoomed in version of the late steps:

![1, 7, 71: step, 1500-6000](images/1_7_71_step_1500-6000.png)

- Before step 2200 or so, the MoT is slightly better than MoT-sum, but afterwards, no difference is visible.
- Both follow the same strange shape, so it isn't *extremely* architecture dependent (though they are of course very similar)

Let's zoom in further at the end of the plot:

![1, 7, 71: step, 5400-6000](images/1_7_71_step_5400-6000.png)

- The two are very close for a while, then diverge again
- The MoT is better than the MoT-sum, but the difference is tiny (final loss MoT: 2.920585, MoT-sum: 2.920994 &rarr; MoT-sum has a final validation loss 1.00014 times larger than MoT, so 0.014% higher)

Let's look at timing:

![1, 7, 71: time](images/1_7_71_time_full.png)

- MoT-sum is significantly faster than MoT
  - This is especially pronounced in the beginning
- The baseline is still faster than the MoT-sum

Let's zoom in a bit:

![1, 7, 71: time, 400-1450 ms](images/1_7_71_time_400-1450.png)

- MoT-sum is worse than the baseline the entire time
- But the real issue is again that weird camel bump

## 72_mot-in_toks-valemb

This is changed from 7:

- Removed the individual norms from the token- and byte-embeddings, and only normed after the FC layer that mixes in the bytes
- Lowered the learning rate of the byte_embeddings from 0.3 to 0.1

Looking back (from after experiment 79 which I will get to), I should have disentangled those changes; I might need to look at them at some later point.

Just looking at the time, 72 has a per-step time of 256.28ms; MoT-sum has 255.56ms, MoT 260.98ms. That's confusing; why should anything change vs. MoT? I might have made a mistake and will have to try the two changes individually later.

![1, 7, 72: step, 1500-6000](images/1_7_72_step_1500-6000.png)

- The regular MoT is very slightly better
- But the difference is negliable, so I'll have to repeat the changes separately and properly

## 73_mot-in_toks-valemb

Changed from 71: instead of `norm(byte_embs + token_embs)`, I'm going `norm(byte_embs) + norm(token_embs)`

> Again, I'm writing this down after having done experiment 79, but at this point, I started registering my predictions

- Prediction: will be worse because model cannot itself determine the relative weight of token_embs and byte_embs

![1, 71, 73: step, 1500-6000](images/1_71_73_step_1500-6000.png)

This modification makes performance worse than the original MoT-sum.

## 74_mot-in_toks-valemb

Changed from 73: `norm(byte_embs) * scalars[-1] + norm(token_embs) * scalars[-2]`

- Prediction: will be as good as 71 or better.
  - Issue of relative weight of token_embs and byte_embs is solved
  - But the token_embs and byte_embs themselves still get normed (which seems to have helped with tokens-only)

![1, 71, 74: step, 4500-6000](images/1_71_74_step_4500-6000.png)

My prediction was wrong: this version of MoT-sum is actually worse than 71.

What I haven't tried is `norm( norm(byte_embs) * scalars[-1] + norm(token_embs) * scalars[-2] )`.

## 75_mot-in_toks-valemb

Changed from 72: Reduced token_dim to 896

- Precictions:
  - Faster but worse

Results (since the shape of the plots of 72 and 75 are basically identical and they're very close, I'll just show a zoomed in version), starting with per-step:

![1, 72, 75: step, 5000-6000](images/1_72_75_step_5000-6000.png)

Surprisingly, this is actually better! To me, that points to the Fully Connected layer that projects from the concatenated tokens and bytes into the model dimension being under-tuned. Which makes sense because if `bytes_per_token=16, byte_dim=64, token_dim=1024`, the weight will have shape `1024 x 2048`. That's pretty large (though the expansion factor in the MLPs is also large, so I'm not entirely sure that this makes sense).

That gives me a hint for two next things that I could do:

1. Reduce the dimensions so that the byte-mixin weight has shape `1024 x 1024`; so `byte_dim=32, token_dim=512`
2. Tune the learning rate

Let's look at the time, too:

![1, 72, 75: time, 1250-1500](images/1_72_75_time_1250-1500.png)

The reduced `token_dim` speeds up the training a little bit, which is nice.

These results make me curious about two further comparisons: 1) comparison to 7 (the original MoT), because it's better than 72 and thus a better baseline, and 2) comparison to 71 (the best MoT-sum), because that's also pretty good and very fast (for a MoT).

First off, the comparison to the original MoT:

![1, 7, 75: step, 5000-6000](images/1_7_75_step_5000-6000.png)

The original MoT is slightly better than this one, but I also screwed up and did the hyperparameter tuning at the same time. I'd like to see a comparison between the MoT with reduced `token_dim` but no tuned hyperparameters and the original MoT. Especially because per-time, the comparison looks very different:

![1, 7, 75: time, 1000-1500](images/1_7_75_time_1000-1500.png)

75 is clearly much faster than 7.

Now the comparison to 71, the original MoT-sum:

![1, 71, 75: time, 1000-1500](images/1_71_75_time_1000-1500.png)

Both are equally fast, but MoT-sum is slightly better. However, the difference is tiny and it might be more promising to stick with the normal MoT, for two reasons: 1) I can undo the hyperparameter-tuning that made it worse, and 2) I can further reduce the `token_dim` for the MoT while it's fixed for MoT-sum. The previously proposed MoT-sum variant where I apply a linear layer to the bytes before summing might be worth a try though.
