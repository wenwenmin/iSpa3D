import os
import copy
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm

class RepresentationDataset(Dataset):
    
    def __init__(self, representations, labels):
        if isinstance(representations, np.ndarray):
            self.representations = torch.from_numpy(representations).float()
        else:
            self.representations = representations.float()
            
        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = labels.long()
            
        assert len(self.representations) == len(self.labels), \
            "Representations and labels must have same length"
    
    def __len__(self):
        return len(self.representations)
    
    def __getitem__(self, idx):
        return {
            'x': self.representations[idx],
            'y': self.labels[idx]
        }


class StackMLPModule(nn.Module):
    
    def __init__(self, in_features, n_classes, hidden_dims=[64, 32], dropout=0.2):
        super(StackMLPModule, self).__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        
        # Build MLP layers
        self.layers = nn.ModuleList()
        mlp_dims = [in_features] + hidden_dims
        
        for i in range(len(mlp_dims) - 1):
            self.layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            self.layers.append(nn.BatchNorm1d(mlp_dims[i + 1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
        
        # Output layer
        self.output_layer = nn.Linear(mlp_dims[-1], n_classes)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        logits = self.output_layer(x)
        probs = F.softmax(logits, dim=1)
        
        return {'logits': logits, 'probs': probs}


class iSpaNetClassifier:
    
    def __init__(self, 
                 in_features,
                 n_classes,
                 config,
                 device='cuda'):
        # Load parameters from config
        classifier_config = config.get('classifier', {})
        mlp_config = classifier_config.get('mlp', {})
        model_config = classifier_config.get('model', {})
        optimizer_params = mlp_config.get('optimizer', {}).get('params', {})
        scheduler_params = mlp_config.get('scheduler', {}).get('params', {})
        trainer_config = classifier_config.get('classifier_trainer', {})
        
        # Extract all parameters from config
        hidden_dims = model_config.get('hidden_dims', [64, 32])
        dropout = model_config.get('dropout', 0.2)
        batch_size = model_config.get('batch_size', 128)
        learning_rate = optimizer_params.get('lr', 0.001)
        weight_decay = optimizer_params.get('weight_decay', 0.001)
        scheduler_step_size = scheduler_params.get('step_size', 250)
        scheduler_gamma = scheduler_params.get('gamma', 0.5)
        
        # Store config and training parameters
        self.config = config
        self.in_features = in_features
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.test_prop = mlp_config.get('test_prop', 0.1)
        self.balanced = mlp_config.get('balanced', True)
        self.max_epochs = trainer_config.get('max_epochs', 300)
        self.encoder_lr = classifier_config.get('encoder_lr', 1e-5)
        
        # Build model
        self.model = StackMLPModule(in_features, n_classes, hidden_dims, dropout)
        self.model = self.model.to(device)
        
        # Optimizer for classifier
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.encoder_optimizer = None
        self.encoder_scheduler = None
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=scheduler_step_size, 
            gamma=scheduler_gamma
        )
        
        # Store G3net model reference for fine-tuning
        self.g3net_model = None
        self.edge_index = None
        
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.train_indices = None
        
        self.train_history = {
            'loss': [], 'acc': []
        }
        self.val_history = {
            'loss': [], 'acc': []
        }
        self.confusion_matrix = None
        self.best_model_state = None
    
    def prepare_data(self, 
                     g3net_model,
                     adata,
                     target_labels,
                     test_prop=None,
                     balanced=None):
        """准备训练数据，使用聚类标签训练分类器。
        
        Args:
            g3net_model: 训练好的G3net模型
            adata: AnnData对象
            target_labels: 聚类标签数组（必需）
            test_prop: 验证集比例
            balanced: 是否使用平衡采样
        """
        test_prop = test_prop if test_prop is not None else self.test_prop
        balanced = balanced if balanced is not None else self.balanced
        
        self.g3net_model = g3net_model
        self.edge_index = g3net_model.adj_norm
        
        g3net_model.autoencoder.train()
        
        self.encoder_optimizer = torch.optim.Adam(
            g3net_model.autoencoder.parameters(),
            lr=self.encoder_lr,
            weight_decay=self.weight_decay
        )
        self.encoder_scheduler = torch.optim.lr_scheduler.StepLR(
            self.encoder_optimizer,
            step_size=250,
            gamma=0.5
        )
        
        # Convert labels
        if isinstance(target_labels, np.ndarray):
            target_labels = torch.from_numpy(target_labels).long()
        elif isinstance(target_labels, torch.Tensor):
            target_labels = target_labels.long()
        else:
            raise TypeError("target_labels must be numpy array or torch tensor")
        
        self.all_labels = target_labels
        
        node_indices = torch.arange(len(target_labels))
        full_dataset = RepresentationDataset(node_indices.unsqueeze(1).float(), target_labels)
        
        n_val = int(len(full_dataset) * test_prop)
        n_train = len(full_dataset) - n_val
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [n_train, n_val]
        )
        
        self.train_indices = self.train_dataset.indices
        self.val_indices = self.val_dataset.indices
        
        if balanced:
            train_indices = self.train_dataset.indices
            train_labels = target_labels[train_indices].numpy()
            
            class_counts = np.bincount(train_labels)
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[train_labels]
            sample_weights = torch.from_numpy(sample_weights).double()
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler
            )
        else:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
    
    def train(self, epochs=100, verbose=True, early_stop_patience=20):
        if self.train_loader is None:
            raise RuntimeError("Please call prepare_data() first!")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch()
            self.train_history['loss'].append(train_loss)
            self.train_history['acc'].append(train_acc)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch()
            self.val_history['loss'].append(val_loss)
            self.val_history['acc'].append(val_acc)
            
            self.scheduler.step()
            self.encoder_scheduler.step()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1:3d}/{epochs}] "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
            
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        self._compute_confusion_matrix()
    
    def _train_epoch(self):
        self.model.train()
        self.g3net_model.autoencoder.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in self.train_loader:
            y = batch['y'].to(self.device)
            node_idx = batch['x'].squeeze().long()
            
            latent_full = self.g3net_model.autoencoder.embeding(
                self.g3net_model.X, 
                self.edge_index
            )
            x = latent_full[node_idx]
            
            self.encoder_optimizer.zero_grad()
            self.optimizer.zero_grad()
            
            output = self.model(x)
            loss = F.cross_entropy(output['logits'], y)
            
            loss.backward()
            
            self.encoder_optimizer.step()
            self.optimizer.step()
            
            total_loss += loss.item() * y.size(0)
            pred = output['logits'].argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        
        avg_loss = total_loss / total
        avg_acc = correct / total
        
        return avg_loss, avg_acc
    
    def _validate_epoch(self):
        self.model.eval()
        self.g3net_model.autoencoder.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                y = batch['y'].to(self.device)
                node_idx = batch['x'].squeeze().long()
                
                latent_full = self.g3net_model.autoencoder.embeding(
                    self.g3net_model.X,
                    self.edge_index
                )
                x = latent_full[node_idx]
                
                output = self.model(x)
                loss = F.cross_entropy(output['logits'], y)
                
                total_loss += loss.item() * y.size(0)
                pred = output['logits'].argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        avg_loss = total_loss / total
        avg_acc = correct / total
        
        return avg_loss, avg_acc
    
    def _compute_confusion_matrix(self):
        self.model.eval()
        self.g3net_model.autoencoder.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                y = batch['y']
                node_idx = batch['x'].squeeze().long()
                latent_full = self.g3net_model.autoencoder.embeding(
                    self.g3net_model.X,
                    self.edge_index
                )
                x = latent_full[node_idx].to(self.device)
                
                output = self.model(x)
                pred = output['logits'].argmax(dim=1).cpu()
                
                all_preds.append(pred)
                all_labels.append(y)
        
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        self.confusion_matrix = confusion_matrix(all_labels, all_preds)
        
        return self.confusion_matrix
    
    def predict(self, representations):
        self.model.eval()
        
        if isinstance(representations, np.ndarray):
            representations = torch.from_numpy(representations).float()
        
        representations = representations.to(self.device)
        
        with torch.no_grad():
            output = self.model(representations)
            pred = output['logits'].argmax(dim=1).cpu().numpy()
        
        return pred
    
    def predict_proba(self, representations):
        self.model.eval()
        
        if isinstance(representations, np.ndarray):
            representations = torch.from_numpy(representations).float()
        
        representations = representations.to(self.device)
        
        with torch.no_grad():
            output = self.model(representations)
            probs = output['probs'].cpu().numpy()
        
        return probs
    
    def save_checkpoint(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history,
            'confusion_matrix': self.confusion_matrix,
            'config': {
                'in_features': self.in_features,
                'n_classes': self.n_classes,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay
            }
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        self.confusion_matrix = checkpoint.get('confusion_matrix', None)