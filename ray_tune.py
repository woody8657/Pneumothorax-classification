from ray import tune  
from train import train
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
analysis = tune.run(
    train,
    config={
        'model': tune.grid_search(['inceptionv3', 'regnet', 'resnext']),
        'loss': tune.grid_search(['CE']),
        'optimizer': tune.grid_search(['AdamW']),
        'scheduler': tune.grid_search(['CosineAnnealingLR']),
        'img_size': tune.grid_search([1024]),
        'batch_size': tune.grid_search([32]),
        'epochs': tune.grid_search([1]),
        'lr': tune.grid_search([0.001])
    }, 
    resources_per_trial={"cpu" : 4, "gpu": 1},
    raise_on_failed_trial=False
    )

print("Best config: ", analysis.get_best_config(
    metric="AUC_valid", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df


