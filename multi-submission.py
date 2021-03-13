import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


multisub = pd.read_csv('../input/lyft-motion-prediction-autonomous-vehicles/multi_mode_sample_submission.csv')
cols = list(multisub.columns)
confs = cols[2:5]
conf0 = cols[5:105]
conf1 = cols[105:205]
conf2 = cols[205:305]

!ls ../input

sub0 = pd.read_csv('../input/lyft-submission/submission.csv')
sub1 = pd.read_csv('../input/pytorch-baseline-inference/submission.csv')
sub2 = pd.read_csv('../input/lyft-constant-velocity-extrapolation-baseline/submission.csv')
multisub[confs] = [.5,.15,.35]
multisub[conf0] = sub0[conf0]
multisub[conf1] = sub1[conf0]
multisub[conf2] = sub2[conf0]

multisub.to_csv('submission.csv', index=False, float_format='%.6g')