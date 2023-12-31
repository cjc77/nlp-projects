import numpy as np
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from sklearn.metrics import mean_squared_error
from torch.optim import AdamW


class TextRegressor(nn.Module):
    def __init__(self, embedder: nn.Module, embed_dim: int, pooling_fn: Callable, output_dim: int = 1):
        """A regression model that predicts real-numbered values given tokenized text passages.

        Args:
            embedder (torch.Module): The embedding model that will be used to convert tokens into 
                embeddings.
            embed_dim (int): The dimensionality of the embeddings.
            pooling_fn (Callable): The pooling function that will be used to convert the token 
                embeddings into full sequence embeddings.
            output_dim (int, optional): The output dimension for the regression task.
                Defaults to 1.
        """
        super().__init__()
        
        # Initialize the encoder
        self.embedder = embedder
        
        # Regression head
        self.regression_head = nn.Linear(embed_dim, output_dim)

        # Pooling strategy
        self.pooling_fn = pooling_fn
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Predicts the regression target(s) using input token IDs and their corresponding attention masks.

        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Corresponding attention masks for the input token IDs.

        Returns:
            torch.Tensor: Predictions for the regression target(s).
        """
        # Forward pass through encoder
        embedding = self.embedder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        embedding = self.pooling_fn(embedding, attention_mask)
        
        # Forward pass through regression head
        yhat = self.regression_head(embedding)
        
        return yhat


class LitTextRegressor(pl.LightningModule):
    def __init__(self, text_regressor: TextRegressor):
        super().__init__()
        self.text_regressor = text_regressor
        # Loss
        self.criterion = F.mse_loss
        # This will change with freezing/un-freezing
        self.lr = 1e-3

        self.val_epoch_out = {
            "yhat": [],
            "y": []
        }

        self.val_epoch_metrics = {
            "mse": None,
            "rmse": None
        }

        self.test_epoch_out = {
            "yhat": [],
            "y": []
        }

        self.test_epoch_metrics = {
            "mse": None,
            "rmse": None
        }

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        yhat = self.text_regressor(input_ids=input_ids, attention_mask=attention_mask)
        return yhat

    def training_step(self, batch, batch_idx: int):
        input_ids, attention_mask, ratings = batch
        yhat = self.text_regressor(input_ids=input_ids, attention_mask=attention_mask).view(-1)

        # loss = self.criterion(yhat, ratings.unsqueeze(1))
        loss = self.criterion(yhat, ratings)
        self.log("avg_train_loss", loss.item(), on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        input_ids, attention_mask, ratings = batch
        yhat = self.text_regressor(input_ids=input_ids, attention_mask=attention_mask).view(-1)

        # loss = self.criterion(yhat, ratings.unsqueeze(1))
        loss = self.criterion(yhat, ratings)
        self.log("avg_val_loss", loss.item(), on_epoch=True, on_step=False, prog_bar=True)

        self.val_epoch_out["yhat"].append(yhat)
        self.val_epoch_out["y"].append(ratings)

    def on_validation_epoch_start(self):
        for k in self.val_epoch_out.keys():
            self.val_epoch_out[k] = []
        for k in self.val_epoch_metrics.keys():
            self.val_epoch_metrics[k] = None

    def on_validation_epoch_end(self):
        yhat = torch.concat(self.val_epoch_out["yhat"]).to("cpu").numpy()
        y = torch.concat(self.val_epoch_out["y"]).to("cpu").numpy()

        self.val_epoch_metrics["mse"] = mean_squared_error(y, yhat)
        self.val_epoch_metrics["rmse"] = np.sqrt(self.val_epoch_metrics["mse"])

        loggers = self.trainer.loggers

        for k, v in self.val_epoch_metrics.items():
            for logger in loggers:
                if isinstance(logger, pl.loggers.TensorBoardLogger):
                    logger.experiment.add_scalar(f"val/{k}", v, self.current_epoch)
                else:
                    logger.experiment.log_metrics({f"val_{k}": v}, self.current_epoch)

    def test_step(self, batch, batch_idx: int):
        input_ids, attention_mask, ratings = batch
        yhat = self.text_regressor(input_ids=input_ids, attention_mask=attention_mask).view(-1)
        
        loss = self.criterion(yhat, ratings)
        self.log("avg_test_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

        self.test_epoch_out["yhat"].append(yhat)
        self.test_epoch_out["y"].append(ratings)
        
    def on_test_epoch_start(self):
        for k in self.test_epoch_out.keys():
            self.test_epoch_out[k] = []
        for k in self.test_epoch_metrics.keys():
            self.test_epoch_metrics[k] = None

    def on_test_epoch_end(self):
        yhat = torch.concat(self.test_epoch_out["yhat"]).to("cpu").numpy()
        y = torch.concat(self.test_epoch_out["y"]).to("cpu").numpy()

        self.test_epoch_metrics["mse"] = mean_squared_error(y, yhat)
        self.test_epoch_metrics["rmse"] = np.sqrt(self.test_epoch_metrics["mse"])

        loggers = self.trainer.loggers

        for k, v in self.test_epoch_metrics.items():
            for logger in loggers:
                if isinstance(logger, pl.loggers.TensorBoardLogger):
                    logger.experiment.add_scalar(f"test/{k}", v, self.current_epoch)
                else:
                    logger.experiment.log_metrics({f"test_{k}": v}, self.current_epoch)

    def configure_optimizers(self):
        no_wd_parameters = ["word_embeddings", "position_embeddings"]
        # Don't use weight decay on the embedding parameters.
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.text_regressor.named_parameters() if any(excl in n for excl in no_wd_parameters)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.text_regressor.named_parameters() if all(excl not in n for excl in no_wd_parameters)],
                "weight_decay": 0.01,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        return optimizer
        
    def freeze_pretrained_model(self, lr: float = 1e-3):
        """Freeze the weights of the pre-trained model.

        Also, set a new learning rate for the regression head parameters.

        Args:
            lr (float, optional): New learning rate. Defaults to 1e-3.
        """
        for param in self.text_regressor.embedder.parameters():
            param.requires_grad = False
        self.lr = lr

    def unfreeze_pretrained_model(self, lr: float = 1e-5):
        """Un-freeze the weights of the pre-trained model.

        Also, set a new learning rate for the full set of parameters.

        Args:
            lr (float, optional): New learning rate. Defaults to 1e-5.
        """
        for param in self.text_regressor.embedder.parameters():
            param.requires_grad = True
        self.lr = lr
