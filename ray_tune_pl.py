from ray import tune
import pytorch_lightning as pl
from main_save import LitModel
from ray.tune.stopper import TrialPlateauStopper
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import os
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hebo import HEBOSearch
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.bayesopt import BayesOptSearch


def wrap(config):
    train_report = TuneReportCallback({
        'train/loss': 'train/loss',
        'train/acc': 'train/acc',
        'train/AUC': 'train/AUC',
        'val/loss': 'val/loss',
        'val/acc': 'val/acc',
        'val/AUC': 'val/AUC',
        'lr': 'lr',
        'best_epoch': 'best_epoch', 
        "hp_metric": "hp_metric"
    },on="validation_end")
        
    trainer = pl.Trainer(
        # logger=None,
        gpus=1,
        max_epochs=config['epochs'],
        # flush_logs_every_n_steps=1,
        check_val_every_n_epoch=1,
        callbacks=[
            train_report
        ],
        overfit_batches=config['overfit'] 
    )
    
    trainer.fit(LitModel(config))
    trainer.save_checkpoint('final.ckpt')
    


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["SLURM_JOB_NAME"] = 'bash'
# os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '0'

# hebo
# previously_run_params = [
#         {
#         "batch_size": 48,
#         "epochs": 100,
#         "img_size": 512,
#         "loss": "CE",
#         "lr": 0.00028542738292130213,
#         "model": "inceptionv3",
#         "optimizer": "AdamW",
#         "scheduler": "CosineAnnealingLR",
#         "weight_decay": 0.01034803569256613
#         },
#         {
#         "batch_size": 48,
#         "epochs": 100,
#         "img_size": 512,
#         "loss": "CE",
#         "lr": 0.0004667399595416,
#         "model": "inceptionv3",
#         "optimizer": "AdamW",
#         "scheduler": None,
#         "weight_decay": 0.01763383003982534
#         },
#         {
#         "batch_size": 48,
#         "epochs": 100,
#         "img_size": 512,
#         "loss": "CE",
#         "lr": 0.0005623413251903496,
#         "model": "inceptionv3",
#         "optimizer": "AdamW",
#         "scheduler": None,
#         "weight_decay": 0.03162277660168381
#         }
#     ]
# known_rewards = [0.9337512254714966, 0.9244714975357056, 0.924963116645813]

# algo = HEBOSearch(points_to_evaluate=previously_run_params,
#     evaluated_rewards=known_rewards,
#     metric="hp_metric",
#     mode="max")
# scheduler = AsyncHyperBandScheduler()

# baysian
algo = BayesOptSearch(utility_kwargs={
        "kind": "ucb",
        "kappa": 2.5,
        "xi": 0.0
    })
algo = ConcurrencyLimiter(algo, max_concurrent=4)
scheduler = AsyncHyperBandScheduler()


analysis = tune.run(
    wrap,
    local_dir='ray_logs/Bayseopt',
    resources_per_trial={ "cpu": 4,"gpu": 1},
    metric="hp_metric",
    mode="max",
    # search_alg=algo,
    # scheduler=scheduler,
    verbose=0, 
    config={
        'model': 'resnext',
        'loss': 'CE',
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingLR',
        'img_size': 512,
        'batch_size': 20,
        'epochs': 100,
        'lr': tune.loguniform(1e-4, 1e-3),
        'weight_decay': tune.loguniform(1e-1, 1e-3),
        'overfit': 100
    },
    # stop=TrialPlateauStopper(metric='val/AUC', std=0.00025, num_results=15),
    num_samples=1
)

print("Best config: ", analysis.get_best_config(
    metric="hp_metric", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df
df.to_csv("result.csv")

