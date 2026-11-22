from abc import ABC, abstractmethod
import copy
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from helpers.metrics import compute_deferral_metrics
from helpers.utils import AverageMeter, accuracy


class BaseMethod(ABC):
    """Abstract method for learning to defer methods"""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        """this function should fit the model and be enough to evaluate the model"""
        pass

    @abstractmethod
    def test(self, dataloader):
        """this function should return a dict with the following keys:
        'defers': deferred binary predictions
        'preds':  classifier predictions
        'labels': labels
        'hum_preds': human predictions
        'rej_score': a real score for the rejector, the higher the more likely to be rejected
        'class_probs': probability of the classifier for each class (can be scores as well)
        """
        pass


class BaseSurrogateMethod(BaseMethod):
    """Abstract method for learning to defer methods based on a surrogate model"""

    def __init__(self, alpha, plotting_interval, model, device, learnable_threshold_rej = False):
        '''
        alpha: hyperparameter for surrogate loss 
        plotting_interval (int): used for plotting model training in fit_epoch
        model (pytorch model): model used for surrogate
        device: cuda device or cpu
        learnable_threshold_rej (bool): whether to learn a treshold on the reject score (applicable to RealizableSurrogate only)
        '''
        self.alpha = alpha
        self.plotting_interval = plotting_interval
        self.model = model
        self.device = device
        self.threshold_rej = 0
        self.learnable_threshold_rej = learnable_threshold_rej 

    @abstractmethod
    def surrogate_loss_function(self, outputs, hum_preds, data_y):
        """surrogate loss function"""
        pass

    def fit_epoch(self, dataloader, optimizer, verbose=False, epoch=1, warmup_iters=0, lrate=1e-3):
        """
        Fit the model for one epoch
        model: model to be trained
        dataloader: dataloader
        optimizer: optimizer
        verbose: print loss
        epoch: epoch number
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        self.model.train()
        
        for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
            if epoch < warmup_iters:
                lr_ = lrate * float(epoch) / warmup_iters
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)
            hum_preds = hum_preds.to(self.device)
            outputs = self.model(data_x)
            loss = self.surrogate_loss_function(outputs, hum_preds, data_y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss ({loss.item()}) at epoch {epoch}, batch {batch}.")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1 = accuracy(outputs.data, data_y, topk=(1,))[0]
            losses.update(loss.data.item(), data_x.size(0))
            top1.update(prec1.item(), data_x.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if verbose and batch % self.plotting_interval == 0:
                logging.info(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        epoch,
                        batch,
                        len(dataloader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                    )
                )

    def fit(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        epochs,
        optimizer,
        lr,
        scheduler=None,
        verbose=True,
        test_interval=5,
        warmup_iters=0,
    ):
        """Train for the requested epochs and keep the final weights if no validation loader is given.

        A validation loader enables the original checkpoint selection and optional threshold
        fitting. A test loader requests metrics for the resulting model.
        """
        if self.learnable_threshold_rej and dataloader_val is None:
            raise ValueError("Learning a rejector threshold requires a validation loader.")
        optimizer = optimizer(self.model.parameters(), lr=lr)
        if scheduler is not None:
            scheduler = scheduler(optimizer, len(dataloader_train) * epochs)
        best_acc = 0
        best_model = copy.deepcopy(self.model.state_dict()) if dataloader_val is not None else None
        for epoch in tqdm(range(epochs)):
            self.fit_epoch(dataloader_train, optimizer, verbose, epoch, warmup_iters=warmup_iters, lrate=lr)
            if dataloader_val is not None and epoch % test_interval == 0 and epoch > 1:
                if self.learnable_threshold_rej:
                    self.fit_treshold_rej(dataloader_val)
                data_test = self.test(dataloader_val)
                val_metrics = compute_deferral_metrics(data_test)
                if val_metrics["system_acc"] >= best_acc:
                    best_acc = val_metrics["system_acc"]
                    best_model = copy.deepcopy(self.model.state_dict())
                if verbose:
                    logging.info(val_metrics)
            if scheduler is not None:
                scheduler.step()
        if best_model is not None:
            self.model.load_state_dict(best_model)
        if self.learnable_threshold_rej:
            self.fit_treshold_rej(dataloader_val)
        if dataloader_test is None:
            return
        final_test = self.test(dataloader_test)
        return compute_deferral_metrics(final_test)

    def fit_treshold_rej(self, dataloader):
        data_test = self.test(dataloader)
        rej_scores = np.unique(data_test["rej_score"])
        rej_scores_quantiles = np.quantile(rej_scores, np.linspace(0, 1, 100))
        best_treshold = 0
        best_accuracy = 0
        for q in rej_scores_quantiles:
            # get deferral decision
            defers = (data_test["rej_score"] > q).astype(int)
            copy_data = copy.deepcopy(data_test)
            copy_data["defers"] = defers
            # compute metrics
            metrics = compute_deferral_metrics(copy_data)
            if metrics['system_acc'] > best_accuracy:
                best_accuracy = metrics['system_acc']
                best_treshold = q
        self.threshold_rej = best_treshold
        
    
    def test(self, dataloader):
        """
        Test the model
        dataloader: dataloader
        """
        defers_all = []
        truths_all = []
        hum_preds_all = []
        predictions_all = []  # classifier only
        rej_score_all = []  # Base routing margin: expert score minus the best class score.
        class_probs_all = []  # classifier probability
        output_g_all = []
        rej_prob_all = []
        self.model.eval()
        with torch.no_grad():
            for data_x, data_y, hum_preds in dataloader:
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                hum_preds = hum_preds.to(self.device)

                outputs = self.model(data_x)
                output_g_all.extend(outputs.detach().cpu().numpy())
                outputs_class = F.softmax(outputs[:, :-1], dim=1)
                outputs = F.softmax(outputs, dim=1)
                predicted_class = outputs[:, :-1].argmax(dim=1)
                predictions_all.extend(predicted_class.cpu().numpy())
                
                expert_scores = outputs[:, -1].tolist()
                classifier_scores = outputs.gather(1, predicted_class[:, None]).flatten().tolist()
                margins = [expert - classifier for expert, classifier in zip(expert_scores, classifier_scores)]
                rej_score_all.extend(margins)
                rej_prob_all.extend(expert_scores)
                defers_all.extend(int(margin >= self.threshold_rej) for margin in margins)
                truths_all.extend(data_y.cpu().numpy())
                hum_preds_all.extend(hum_preds.cpu().numpy())
                class_probs_all.extend(outputs_class.cpu().numpy())

        return {
            "defers": np.array(defers_all),
            "labels": np.array(truths_all),
            "hum_preds": np.array(hum_preds_all),
            "preds": np.array(predictions_all),
            "rej_score": np.array(rej_score_all),
            "class_probs": np.array(class_probs_all),
            "output_g_all": np.array(output_g_all),
            "rej_prob_all": np.array(rej_prob_all),
        }
