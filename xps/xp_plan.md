## Data spec

Use VB+DMD at 16khz, use speakers p226 and p287 for validation
## Data efficiency

This exp aims to see how robust the models are to limited data. configure splits by removing speakers, consider a number of splits e.g.: full, medium, low, very low, single speaker.

use sgmse, sb-ve, and icfm models, use c=0.1 for icfm, k=2.6 and c=0.4 for sb, and use a good k and c for sgmse. 

## Conditioning ablation

all these models condition on the original noisy input, make a model which is not conditioned on the noisy input, only on the current sample and time index. See how sgmse, sb-ve, and icfm perform on the full data split without conditioning, my hypothesis is that icfm will be robust to this but the others won't, due to the less noisy training objective.

## Direct data baseline
We should have two baselines, one icfm that trains only with t=1 (ddp only), and then one unet which isn't conditioned by t, a simple noise to clean regressor

## augmented DDP

DDP has proven to be good, what if we finetune an ICFM model with the DDP task, or simulate annealing of p(t=1) during training so it starts with t=[0,1] uniformly then gradually becomes a ddp predictor when p(t=1)=1.
