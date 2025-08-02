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

So it might be some sort of threshold being reached for the learning rate, but it's still strange, especially because the final loss is very close (baseline: 2.919627, MoT: 2.920585 &rarr; the MoT loss is 1.00032812 times larger than the baseline loss, or ~0.033%).

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

Looking back again, I was thinking about the strange hump in the loss curve of all MoT-variants again, and making this plan:

- Next steps:
  - [ ] Increase Batch size (I saved a little bit of memory)
  - [ ] Tune hparams, especially for the byte_mixin (Prio 4)
  - [x] Experiment with increasing the un-masked attention window for both the base & the MoT (Prio 3)
  - [x] Change lr schedule (Prio 2)
    - warmup_frac = 1 - cooldown_frac
    - Do WD instead of SD as it is now (and not WSD, this ain't production material)
  - [x] Shuffle the data; maybe there's a problem where there are a bunch of tokens that suck for the MoT and that's why the curve bends so strangely? Actually that should be theory number 1, because neither the (Prio 1)

## 01_train_gpt_medium

Changed from 0_train_gpt_medium:

- Changed from SD schedule to WD schedule
- Increased cooldown_frac from 0.7 to 0.95
- warmup_frac = 1 - cooldown_frac

Expectations:

- I'm really 50/50 if it will work better or worse than the original
  - On the one hand, I read that WD is strictly better than SD
  - On the other hand, modded-nanogpt is pretty well tuned already, so who knows?
  - Also, I don't know if the WD results hold for Muon at all
- It might work better for the MoT, if the learning rate is the cause of the strange break in the loss curve of the MoT variants

(The above are my original notes, below are the ones from after 79)

I was a bit dumb doing this comparison starting from 0_train_gpt_medium instead of 1_toks-in_toks-valemb, but since the experiments were a failure anyway it's fine

![0, 01: step](images/0_01_step_full.png)

The adjusted learning rate schedule makes the run worse; but it's especially interesting *how* it does it. First, 01 is a lot worse than 0, then it's slightly better, and then noticably worse again. That's like a slightly different version of the MoT-loss-curves! That's a big sign that the issue with the MoT is learning-rate related.

I won't make the comparison over time because the difference is tiny.

## 76_mot-in_toks-valemb

Changed from 75_mot-in_toks-valemb:

- Same learning rate schedule as [01_train_gpt_medium](#01_train_gpt_medium)

![75, 76: time](images/75_76_time_full.png)

Clearly, this learning rate schedule makes things worse for the MoT, too.

## 02_train_gpt_medium

What I originally wrote:

Changed from 0_train_gpt_medium:

- Randomly shuffled files (with fixed seed)

Expectations:

- I think that this likely won't change much

My thought process was that the strange hump might be caused by some kind of data issue; the dataloader in modded-nanogpt loads one file after the other, so the thought was that it would load a file with data that worked well, then one with data that has some issue, and then another good one (the data order is fixed). To find out, I shuffled the training data. The actual target is the MoT (which will come below), this run was just for comparison (in case the data-shuffling caused some sort of change in both runs). Here you can see the results over time:

![0, 02: time](images/0_02_time_full.png)

Barely a difference is visible.

![0, 02: time, 1200-1500](images/0_02_time_1200-1500.png)

Zoomed in, we can see that the shuffled data actually underperforms the original data order (I assume that that's just random chance; after all, the model initialization is also changed).

## 77_mot-in_toks-valemb

What I origianlly wrote:

Changed from 75_mot-in_toks-valemb:

- Randomly shuffled files (with fixed seed)

Expectations:

- I'm at ~80% that this will significantly change the shape of the loss curve
- That's because my leading theory for why the plot curve looks like shit is some data problem (that I should look into later)

And here are the results:

![0, 75, 77: time, 1100-1500](images/0_75_77_time_1100-1500.png)

Shuffling changes absolutely nothing for the MoT except for random noise.

After this surprising (to me) result, I decided to do two things:

1. Test if the issue is related to the sequence length
    - I had previously trained a MoT on ~50B tokens
    - It had significantly lower training and validation loss than the token-only baseline
    - It was modified from an older version of modded-nanogpt (medium)
    - Specifically, the sliding window mask was removed
    - So I thought that that might be the issue
    - And I wanted to test increasing the sequence length earlier just to see what the effect would be (not the most rigorous test, but easy to do and a decent proxy for whether or not this idea was promising)
2. Check a few other statistics
    - First off, making sure that the files were actually shuffled
    - Secondly, checking the number of total bytes since the last validation step, as well as the number of bytes that were pulled and the number of bytes that were used to block context from another document in the same sequence. This was to check out a potential data-statistic that is unique to the MoT and might differ over the course of training

## 03_mot-in_toks-valemb

First off, changing the sequence-length (I changed the schedule; looking back, I don't know why I didn't just choose the maximum sequence length the entire time instead of a weird schedule, but whatever, this is still valuable data). Here's what I wrote:

Changed from 00:

- Changed seq-len schedule to `math.sqrt(x * (2 - x))`

Expectations:

- Lower val-loss than 00 (slightly)
- Slower than 00 (significantly)

Here is the plot of the updated sequence-length schedule, compared to the default (cubic) schedule:

![learning rate schedule: cubic vs. square root](images/lr_schedule_cubic_and_sqrt.png)

The new sequence length is always higher than the default one except in the very beginning and end. So what are the results?

![0, 03: step, 100-6000](images/0_03_step_100-6000.png)

Above is almost the entire plot (only the first few step are cut off). It shows that the new schedule learns more over almost the entire run (per step, not per time-step; I'll come to that). However, that breaks down in the very end. Let's look more closely at that:

![0, 03: step, 5000-6000](images/0_03_step_5000-6000.png)

The run with a higher sequence length throughout training does worse than the original one. Huh? I guess this is mostly a question of hyperparameter tuning (but also, I should try a modded-nanogpt run going in the other direction; maybe a constant low sequence length followed by the original cubic schedule but in shorter time?).

Just as a sanity check, let's look at the time, too:

![0, 03: time, 1000-1500](images/0_03_time_1000-1500.png)

Yeah, the faster sequence length growth makes the run way slower (as expected) and worse. Now, most likely the MoT will also be worsened by this new sequence length schedule. But will it remove the loss hump?

## 78_mot-in_toks-valemb

What I wrote:

Changed from 75:

- Changed seq-len schedule to `math.sqrt(x * (2 - x))`

Expectations:

- Lower val-loss than 75
- Slower than 75
- Interesting is comparison to normal modded-nanogpt but with the new schedule

![1, 75, 78: step, 1500-6000](images/1_75_78_step_1500-6000.png)

The new sequence length schedule changes nothing about the loss hump, but it does reduce final performance of the model. This is unlikely to be the reason for the strange loss hump.

So the last thing I did was capture a few statistics.

## 79_mot-in_toks_valemb

Changed from 77:

- Log current file (from dataloader), total_bytes, pulled_bytes, blocked_bytes

![79-byte-stats](images/79-byte-stats.png)

- All the byte stats are very consistent (the lower values at the start and end are just because the number of steps in between is lower, which I didn't correct for)
- The number of pulled bytes is consistently at slightly above 70% of the total bytes
- The number of blocked bytes is around 0.0025%
- The files are actually shuffled
- So my favorite hypothesis of something being wrong with the data is disproven; which is very good in a sense, but bad in another because now I don't know what to fix

## NEXT STEPS (2025-07-25)

Here is the experiments I had planned before creating this document:

- [ ] Do MoT by addition, but apply linear layer to the bytes before so that they can be mixed
- [ ] Zero-init bytes?
- [ ] Try it with `bpt=8, byte_dim=128` and `bpt=32, byte_dim=32`
  - So many bytes are pulled that it might hurt
  - On the other hand, doing it more might give more of an advantage
- [ ] Decrease `token_dim` (and maybe `byte_dim` too?) but increase expansion factor in MLP.
- [x] `token_dim=512, byte_dim=32`; then only concatenate, no sum or FC layer
  - This should lead to a very clean gradient to both the token- and byte-embeddings
  - In a sense, it lets the actual transformer backend handle the byte mixin.

Now, I would add the following:

- [ ] Independently test norm changes and hyperparameter changes
- [ ] This isn't really a specific TODO, but I should note that the normal MoT with reduced `token_dim`, worked really well so far

The plan:

- Add `num_params` to the end-printout
- Runs:
  - 711: MoT by concatenation (`token_dim=512, byte_dim=32`)
  - 712: If 711 works well, increase the expansion factor of the MLP until the number of parameters is similar to the baseline again

## 711_mot-in_toks-valemb

MoT by concatenation (`token_dim=512, byte_dim=32`); Hyperparameters: (`lr_tok=0.3, lr_byte=0.1`) -> still likely suboptimal (at least for the default MoT 7, it worsened performance).

Predictions:

- Should be pretty fast; only thing that makes it slower than 1 (baseline with my dataloader) is the concatenation ops
- Should be pretty good; token embeddings of size 512 are not too bad, and the gradient signal isn't blocked in any way, so that alone should be okay; and the bytes should only add to that (oooooh, that would be a nice baseline: replace the bytes with a single learned vector).
- I don't think it will remove the loss hump, simply because I'm now convinced that that's mostly due to a change in optimal hyperparameters (or the fundamental structure of mixed token- and byte-embeddings, but that seems less likely)

Here is the validation loss over the steps, compared to the original MoT (7), the MoT through addition (71), and of course the baseline (1):

![1, 7, 71, 711: step, 2500-6000](images/1_7_71_711_step_2500-6000.png)

711 is very clearly the worst of all the baselines that matter. But maybe it's so much faster that it's still worth looking into?

![1, 7, 71, 711: time, 800-1600](images/1_7_71_711_time_800-1600.png)

711 is actually sufficiently much faster than 7 (MoT) that it would be a good, cheaper alternative. But 71 (MoT-sum) crushes it; it's not only better per step, but also faster. Of course, the baseline is still better in every way.

Alright, so I should dismiss this idea. However, I had run another experiment in parallel: 712.

## 712_mot-in_toks-valemb

This is like [711](#711_mot-in_toks-valemb) but the MLP hidden dimension has been increased by 768. This is motivated by the fact that reducing the `token_dim` reduces the total number of parameters quite significantly. I wanted to make up for that by increasing the MLP hidden dimension and see how it goes.

Number of parameters:

- [Baseline (0)](#0_train_gpt_medium): 454_496_336
- [MoT-concat (711)](#711_mot-in_toks-valemb): 428_779_408
- [MoT-concat with increased hidden dim (712)](#712_mot-in_toks-valemb): 453_945_232

So the total number of parameters still doesn't quite match the baseline. Of course, embedding parameters and MLP parameters aren't perfectly comparable: embeddings are sparsely activated while the MLP works on every token in every batch. A fairer comparison would probably be to increase the number of experts in an MoE while keeping the number of active parameters constant, but I'm not about to implement that (not to mention that it would be a radical architecture change that would make the comparison worse). In fact, even then, the MoE has to be held in GPU memory, while the embeddings can be offloaded to another device, because their inputs are one-hot they can thus be fetched via a simple table lookup instead of

So here is the comparison to the [Baseline](#1_toks-in_toks-valemb) and [MoT-concat](#711_mot-in_toks-valemb) per step:

![1, 711, 712: step, 2500-6000](images/1_711_712_step_2500-6000.png)

And damn! This crushes. However, it still has the strange loss hump, and again: MLP parameters and embedding parameters aren't super comparable. That last fact is very visible when we compare performance over time:

![1, 711, 712: time, 800-1700](images/1_711_712_time_800-1700.png)

It's ridiculously much slower. This is clearly not the way to go.
