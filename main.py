from models.xception import create_model
from trainers.minerl.treechop_expert_baseline import train

model = create_model((64, 64, 3), 8)

train(model)