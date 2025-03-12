"""
Training functions for graph neural networks.

This module provides functions for training graph neural networks on node classification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from typing import Tuple, List, Dict, Union, Optional, Any
import numpy as np
from .metrics import evaluate_metrics


def train_one_epoch(
    g: dgl.DGLGraph,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_mask: torch.Tensor,
    loss_fcn: nn.Module,
    metric_type: str = 'accuracy'
) -> Tuple[float, float]:
    """
    Train a graph neural network for one epoch.
    
    Args:
        g: Input graph
        model: Neural network model
        optimizer: Optimizer for training
        train_mask: Boolean mask indicating training nodes
        loss_fcn: Loss function
        metric_type: Type of metric to compute, either 'accuracy' or 'roc_auc'
        
    Returns:
        Tuple[float, float]: Loss value and training metric
    """
    model.train()
    optimizer.zero_grad()
    logits = model(g, g.ndata['feat'])
    loss = loss_fcn(logits[train_mask], g.ndata['label'][train_mask])
    loss.backward()
    optimizer.step()
    
    # Compute training metric
    with torch.no_grad():
        metric = evaluate_metrics(logits, g.ndata['label'], train_mask, metric_type)
    
    return loss.item(), metric


def validate(
    g: dgl.DGLGraph,
    model: nn.Module,
    val_mask: torch.Tensor,
    metric_type: str = 'accuracy'
) -> float:
    """
    Validate a graph neural network model.
    
    Args:
        g: Input graph
        model: Neural network model
        val_mask: Boolean mask indicating validation nodes
        metric_type: Type of metric to compute, either 'accuracy' or 'roc_auc'
        
    Returns:
        float: Validation metric
    """
    model.eval()
    with torch.no_grad():
        logits = model(g, g.ndata['feat'])
        val_metric = evaluate_metrics(logits, g.ndata['label'], val_mask, metric_type)
    return val_metric


def train(
    g: dgl.DGLGraph,
    model: nn.Module,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    model_lr: float = 1e-3,
    optimizer_weight_decay: float = 0.0,
    n_epochs: int = 1000,
    early_stopping: int = 50,
    log_training: bool = False,
    metric_type: str = 'accuracy'
) -> Tuple[float, float, float, nn.Module]:
    """
    Train a graph neural network with early stopping.
    
    The early stopping criterion compares the current validation loss with the average of the
    previous `early_stopping` validation losses. If the current loss exceeds that average, training
    is halted.
    
    Args:
        g: Input graph
        model: Neural network model to train
        train_mask: Boolean mask indicating training nodes
        val_mask: Boolean mask indicating validation nodes
        test_mask: Boolean mask indicating test nodes
        model_lr: Learning rate for the optimizer
        optimizer_weight_decay: Weight decay for the optimizer
        n_epochs: Maximum number of training epochs
        early_stopping: Number of epochs to look back for early stopping
        log_training: Whether to print training progress
        metric_type: Type of metric to compute, either 'accuracy' or 'roc_auc'
        
    Returns:
        Tuple[float, float, float, nn.Module]:
            - Final training metric
            - Final validation metric
            - Final test metric
            - Trained model
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=model_lr, weight_decay=optimizer_weight_decay
    )
    loss_fcn = nn.CrossEntropyLoss()

    best_val_metric = 0 if metric_type == 'accuracy' else -float('inf')
    best_test_metric = 0
    best_state = None

    # Lists to store validation losses and metrics at each epoch.
    val_loss_history = []
    val_metric_history = []

    for epoch in range(n_epochs):
        # Train for one epoch.
        loss, train_metric = train_one_epoch(g, model, optimizer, train_mask, loss_fcn, metric_type)

        # Evaluate on the validation set.
        model.eval()
        with torch.no_grad():
            logits_val = model(g, g.ndata['feat'])
            current_val_loss = loss_fcn(logits_val[val_mask], g.ndata['label'][val_mask]).item()
            current_val_metric = evaluate_metrics(logits_val, g.ndata['label'], val_mask, metric_type)

        # Store the current validation loss and metric.
        val_loss_history.append(current_val_loss)
        val_metric_history.append(current_val_metric)

        # Update the best model if the current validation metric improves.
        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            best_state = model.state_dict()
            best_test_metric = evaluate_metrics(logits_val, g.ndata['label'], test_mask, metric_type)

        if log_training and (epoch % 10 == 0):
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train {metric_type}: {train_metric:.4f} | "
                  f"Val Loss: {current_val_loss:.4f} | Val {metric_type}: {current_val_metric:.4f}")

        # --- Early Stopping Method ---
        # After accumulating at least (early_stopping+1) epochs, compare the current validation loss
        # with the mean of the previous `early_stopping` losses.
        if early_stopping > 0 and epoch > early_stopping:
            # Get the previous `early_stopping` validation losses (excluding the current one)
            prev_losses = val_loss_history[-(early_stopping + 1):-1]
            prev_loss_mean = sum(prev_losses) / len(prev_losses)
            if current_val_loss > prev_loss_mean:
                if log_training:
                    print("Early stopping triggered.")
                break

    # Load the best model state before testing.
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate the best model on the test set.
    # Final metrics
    model.eval()
    with torch.no_grad():
        logits = model(g, g.ndata['feat'])
        final_train_metric = evaluate_metrics(logits, g.ndata['label'], train_mask, metric_type)
        final_val_metric = evaluate_metrics(logits, g.ndata['label'], val_mask, metric_type)
        final_test_metric = evaluate_metrics(logits, g.ndata['label'], test_mask, metric_type)

    return final_train_metric, final_val_metric, final_test_metric, model


def train_selective(
    g_list: List[dgl.DGLGraph],
    model_selective: nn.Module,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    model_lr: float = 1e-3,
    optimizer_weight_decay: float = 0.0,
    n_epochs: int = 1000,
    early_stopping: int = 50,
    log_training: bool = False,
    metric_type: str = 'accuracy'
) -> Tuple[float, float, float, nn.Module]:
    """
    Train a selective graph neural network on multiple graph versions.
    
    Similar to 'train()', but passes a list of graphs to the model.
    Uses a new early stopping method based on validation loss:
    if the current validation loss is higher than the mean of the previous `early_stopping`
    validation losses, training is halted early.
    
    Args:
        g_list: List of input graphs
        model_selective: Selective neural network model to train
        train_mask: Boolean mask indicating training nodes
        val_mask: Boolean mask indicating validation nodes
        test_mask: Boolean mask indicating test nodes
        model_lr: Learning rate for the optimizer
        optimizer_weight_decay: Weight decay for the optimizer
        n_epochs: Maximum number of training epochs
        early_stopping: Number of epochs to look back for early stopping
        log_training: Whether to print training progress
        metric_type: Type of metric to compute, either 'accuracy' or 'roc_auc'
        
    Returns:
        Tuple[float, float, float, nn.Module]:
            - Final training metric
            - Final validation metric
            - Final test metric
            - Trained model
    """
    device = next(model_selective.parameters()).device
    optimizer = torch.optim.Adam(
        model_selective.parameters(), lr=model_lr, weight_decay=optimizer_weight_decay
    )
    loss_fcn = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_test_acc = 0
    best_state = None

    # We'll assume the 'main' graph for feature/label usage is g_list[0]
    g_main = g_list[0]
    feat = g_main.ndata['feat']
    labels = g_main.ndata['label']

    # List to store validation losses for early stopping check.
    val_loss_history = []

    for epoch in range(n_epochs):
        # --------------------
        # Training Phase
        # --------------------
        model_selective.train()
        optimizer.zero_grad()
        logits = model_selective(g_list, feat)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        # --------------------
        # Validation Phase
        # --------------------
        model_selective.eval()
        with torch.no_grad():
            logits_eval = model_selective(g_list, feat)
            current_val_loss = loss_fcn(logits_eval[val_mask], labels[val_mask]).item()
            val_acc = evaluate_metrics(logits_eval, labels, val_mask, metric_type)

        # Append current validation loss to history.
        val_loss_history.append(current_val_loss)

        # Update the best model if the current validation accuracy is improved.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model_selective.state_dict()
            best_test_acc = evaluate_metrics(logits_eval, labels, test_mask, metric_type)

        if log_training and (epoch % 10 == 0):
            print(f"[Selective] Epoch {epoch:03d} | Loss {loss.item():.4f} | ValAcc {val_acc:.4f} | ValLoss {current_val_loss:.4f}")

        # --------------------
        # Early Stopping Check
        # --------------------
        if early_stopping > 0 and epoch > early_stopping:
            # Get the previous 'early_stopping' validation losses (excluding the current epoch)
            prev_losses = val_loss_history[-(early_stopping + 1):-1]
            prev_loss_mean = sum(prev_losses) / len(prev_losses)
            if current_val_loss > prev_loss_mean:
                if log_training:
                    print("Early stopping triggered (selective).")
                break

    # Load the best model state before final evaluation.
    if best_state is not None:
        model_selective.load_state_dict(best_state)

    # --------------------
    # Final Evaluation
    # --------------------
    model_selective.eval()
    with torch.no_grad():
        logits_eval = model_selective(g_list, feat)
        final_train_acc = evaluate_metrics(logits_eval, labels, train_mask, metric_type)
        final_val_acc = evaluate_metrics(logits_eval, labels, val_mask, metric_type)
        final_test_acc = evaluate_metrics(logits_eval, labels, test_mask, metric_type)

    return final_train_acc, final_val_acc, final_test_acc, model_selective
