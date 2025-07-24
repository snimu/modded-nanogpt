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
