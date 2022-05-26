from models.xception import create_model
from trainers.minerl.treechop_expert_amiranas import train

model = create_model((64, 64, 3), 112)

train(model)