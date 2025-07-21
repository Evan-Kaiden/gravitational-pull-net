import torch

NUM_AGENTS = 3
NUM_LEADERS = 1
TRAIN_EPOCHS = 1
TUNE_STEPS = 50
NUM_OPTIMIZATIONS = 100
UPDATE_ITERATIONS = 10
IMPROVEMENT_CUTOFF_ITERATIONS = 15
SELECT_LEADER_STEP = 1


if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f'Device: {device}')