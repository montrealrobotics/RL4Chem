import hydra
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sklearn.metrics
from omegaconf import DictConfig
from data import get_data

class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_params(model):
    return (p for p in model.parameters() if p.requires_grad)

def train(cfg):
    #get trainloader
    train_loader, val_loader, vocab, input_size = get_data(cfg, get_max_len=True)

    #get model
    cfg.vocab_size = len(vocab)
    cfg.pad_idx = vocab.pad
    cfg.input_size = int(input_size)
    model = hydra.utils.instantiate(cfg.charconv)
    
    #set optimizer
    optimizer = optim.Adam(get_params(model), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=0.0)

    num_params = count_parameters(model)
    print('The model has ', num_params, ' number of trainable parameters.')

    avg_train_loss = Averager()
    avg_grad_norm = Averager()
    for epoch in range(cfg.num_epochs):
        metrics = dict()
        for step, (x, y) in enumerate(train_loader):
            preds = model(x)
            loss = F.mse_loss(preds, y)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            avg_train_loss.add(loss.item())
            avg_grad_norm.add(grad_norm.item())

            if step % cfg.eval_interval == 0:
                metrics.update(
                    eval(model, val_loader))
                print('Epoch = ', epoch, 'Step = ', step, ' r2_score = ', metrics['r2 score'])
                model.train()
                
            if cfg.wandb_log and step % cfg.log_interval==0:
                metrics['train loss'] = avg_train_loss.item()
                metrics['average grad norm'] = avg_grad_norm.item()
                metrics['lr'] = scheduler.get_last_lr()[0]
                metrics['epoch'] = epoch                
                avg_train_loss = Averager()
                wandb.log(metrics)
        scheduler.step()
    
    metrics.update(
            eval(model, val_loader))
    print('Epoch = ', epoch, 'Step = ', step, ' r2_score = ', metrics['r2 score'])
    
def eval(model, val_loader):
    metrics = dict()
    preds_list = []
    targets_list = []
    avg_loss = Averager()
    model.eval()
    with torch.no_grad():
        for step, (x, y) in enumerate(val_loader):
            preds = model(x)
            preds_list.append(preds)
            targets_list.append(y)
            loss = F.mse_loss(preds, y)
            avg_loss.add(loss.item())

    preds_list = torch.cat(preds_list).tolist()
    targets_list = torch.cat(targets_list).tolist()
    r2_score = sklearn.metrics.r2_score(y_true = targets_list, y_pred = preds_list)
    metrics['r2 score'] = r2_score
    metrics['val loss'] = avg_loss.item()
    return metrics

@hydra.main(config_path='cfgs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    from char_conv import train
    cfg.model_name = 'char_conv'
    if cfg.wandb_log:
        project_name = 'docking-regression-' + cfg.target
        wandb.init(project=project_name, entity=cfg.wandb_entity, config=dict(cfg), dir=hydra_cfg['runtime']['output_dir'])
        wandb.run.name = cfg.wandb_run_name
    train(cfg)
        
if __name__ == '__main__':
    main()