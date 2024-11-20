import numpy as np
import itertools
import torch
from darts.models import ExponentialSmoothing, LightGBMModel, NHiTSModel,TFTModel, BlockRNNModel, TransformerModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.utils import ModelMode, SeasonalityMode

import darts.utils.likelihood_models as Likelihood
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


# Pytorch early stopping rules
my_stopper = EarlyStopping(
    monitor="train_loss",  # Over which values we are optimizing
    patience=10,           # After how many iterations if the loss doesn't improve then it stops optimizing
    min_delta=0.001,       # Round-off error to consider that it didn't improved
    mode="min"
)

cuda_available = torch.cuda.is_available()
print("GPU available: {}".format(cuda_available))
if(cuda_available):
    # device = 'gpu'
    pl_trainer_kwargs1={"callbacks": [my_stopper], "accelerator": 'gpu', "devices": -1}
else:
    # device = 'cpu'
    pl_trainer_kwargs1 = {"callbacks": [my_stopper], "accelerator": "cpu"}


#Return the defined lists of local and global forecasting models
def get_forecasting_models(quantiles): 

    # List of local models
    model_list_local = {
        
        "ExponentialSmoothing": ExponentialSmoothing(
                   seasonal_periods=52, seasonal=SeasonalityMode.ADDITIVE),

        "LightGBM": LightGBMModel(
                lags=[-1,-2,-53], #
                lags_past_covariates=2, 
                lags_future_covariates=[2,2], 
                output_chunk_length=1,
                likelihood="quantile",
                quantiles=quantiles,
                add_encoders={"cyclic": {"past": ["weekofyear","month"],
                                        "future": ["weekofyear","month"]},
                                        'transformer': Scaler() },
                show_warnings=False,
                verbose=-1
            ),

        "TFT": TFTModel(
                input_chunk_length=52, 
                output_chunk_length=1,
                likelihood=Likelihood.QuantileRegression(quantiles=quantiles),
                n_epochs=200,
                batch_size=32,
                dropout=0.2, 
                optimizer_kwargs={'lr': 1e-3}, 
                pl_trainer_kwargs={"callbacks": [my_stopper], "accelerator": "auto", "devices": -1},
                add_encoders={"cyclic": {"past": ["weekofyear","month"],
                                        "future": ["weekofyear","month"]},
                                        'transformer': Scaler() }),

        
        # "NHiTS": NHiTSModel(
        #         input_chunk_length=52,
        #         output_chunk_length=1,
        #         likelihood=Likelihood.QuantileRegression(quantiles=quantiles),
        #         n_epochs=200,
        #         batch_size=32,
        #         num_stacks=2, num_blocks=2, num_layers=2, 
        #         dropout=0.2,
        #         optimizer_kwargs={'lr': 1e-3}, 
        #         pl_trainer_kwargs=pl_trainer_kwargs1,
        #         add_encoders={"cyclic": {"past": ["weekofyear","month"]},'transformer': Scaler() },
        #         show_warnings=False,
        #     ),
    }
    
    # List of global models
    model_list_global = {
        # "LightGBM_Global": LightGBMModel(
        #         lags=[-1,-2,-53], #
        #         lags_past_covariates=2, 
        #         lags_future_covariates=[2,2], 
        #         output_chunk_length=1,
        #         likelihood="quantile",
        #         quantiles=quantiles,
        #         # add_encoders={"cyclic": {"past": ["weekofyear","month"]},'transformer': Scaler() },
        #         add_encoders={"cyclic": {"past": ["weekofyear","month"],
        #                                 "future": ["weekofyear","month"]},
        #                                 'transformer': Scaler() },
        #         show_warnings=False,
        #         verbose=-1
        #     ),
        # "TFT_Global": TFTModel(
        #         input_chunk_length=52, 
        #         output_chunk_length=1,
        #         likelihood=Likelihood.QuantileRegression(quantiles=quantiles),
        #         n_epochs=200,
        #         batch_size=32,
        #         dropout=0.2, 
        #         optimizer_kwargs={'lr': 1e-3}, 
        #         pl_trainer_kwargs={"callbacks": [my_stopper], "accelerator": "auto", "devices": -1},
        #         add_encoders={"cyclic": {"past": ["weekofyear","month"],
        #                                 "future": ["weekofyear","month"]},
        #                                 'transformer': Scaler() })
        # "NHiTS_Global": NHiTSModel(
        #         input_chunk_length=52,
        #         output_chunk_length=1,
        #         likelihood=Likelihood.QuantileRegression(quantiles=quantiles),
        #         n_epochs=200,
        #         batch_size=32,
        #         num_stacks=2, num_blocks=2, num_layers=2, 
        #         dropout=0.2,
        #         optimizer_kwargs={'lr': 1e-3}, 
        #         pl_trainer_kwargs=pl_trainer_kwargs1,
        #         add_encoders={"cyclic": {"past": ["weekofyear","month"]},'transformer': Scaler() },
        #         show_warnings=False,
        #     ),
    }
    return (model_list_local, model_list_global)
