import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from metrics import map_at_N
from matplotlib import pyplot as plt


def get_logits_and_trues_and_loss(model, dataloader, loss_fn=None, device='cpu'):
    loss = 0.0
    model.eval()
    logits, trues = [], []
    losses = []
    for x, y in dataloader:
        with torch.no_grad():
            y_pred = model(x.to(device).float())

        if loss_fn:
            loss = loss_fn(y_pred, y.to(device)).item()
            losses.append(loss)

        logits.extend(y_pred.to('cpu').numpy().tolist())
        trues.extend(y.to('cpu').numpy().tolist())

    if len(losses):
        loss = np.array(losses).mean()

    return np.array(logits), np.array(trues), loss


def evaluate_fn(model, dataloader, loss_fn, device):

    logits, trues, loss = get_logits_and_trues_and_loss(model, dataloader, loss_fn, device=device)

    conf_matrix = confusion_matrix(trues, logits.argmax(1))
    print('\n')
    print(conf_matrix)

    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    fraud_prob = probs[:, 1]
    AUC = roc_auc_score(trues, fraud_prob)

    try:
        tp = conf_matrix[1][1]
    except:
        tp = 0
    try:
        tn = conf_matrix[0][0]
    except:
        tn = 0
    try:
        fp = conf_matrix[0][1]
    except:
        fp = 0
    try:
        fn = conf_matrix[1][0]
    except:
        fn = 0

    ACC = (tp + tn) / (tp + tn + fp + fn)

    return loss, AUC, ACC
