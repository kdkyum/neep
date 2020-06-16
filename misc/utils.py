import os
import shutil

import numpy as np
import torch
from scipy import stats


def save_checkpoint(state, is_best, path):
    filename = os.path.join(path, "checkpoint.pth.tar")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, "model_best.pth.tar"))


def load_checkpoint(opt, model, optimizer):
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        opt.start = checkpoint["start"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(
            "=> loaded checkpoint '{}' (iteration {})".format(
                opt.resume, checkpoint["iteration"]
            )
        )
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))


def logging(i, loss, time_step, preds, train=True):
    tmp = {}
    cum_preds = np.cumsum(preds, axis=1).mean(axis=0)
    pred_rate, _, _, _, _ = stats.linregress(np.arange(len(cum_preds)), cum_preds)
    pred_rate = (1 / time_step) * pred_rate
    tmp["iteration"] = i
    tmp["loss"] = loss
    tmp["pred_rate"] = pred_rate
    if train:
        print("Train  iter: %d  loss: %1.4e  pred: %.5f" % (i, loss, pred_rate))
    else:
        print("Test   iter: %d  loss: %1.4e  pred: %.5f" % (i, loss, pred_rate))

    return tmp


def logging_r(i, loss, time_step, ents, preds):
    tmp = {}
    cum_preds = np.cumsum(preds, axis=1).mean(axis=0)
    pred_rate, _, _, _, _ = stats.linregress(np.arange(len(cum_preds)), cum_preds)
    pred_rate = (1 / time_step) * pred_rate
    tmp["iteration"] = i
    tmp["loss"] = loss
    tmp["pred_rate"] = pred_rate
    _, _, r_value, _, _ = stats.linregress(preds.flatten(), ents.flatten())
    tmp["r_square"] = r_value ** 2
    print("Test   iter: %d  loss: %1.4e  pred: %.5f  R-square: %.5f"
        % (i, loss, pred_rate, r_value ** 2))

    return tmp


def logging_rneep(i, loss, seq_len, preds, train=True):
    tmp = {}
    pred = preds.flatten()
    cum_pred = np.cumsum(pred)
    slope, _, _, _, _ = stats.linregress(
        np.arange(len(cum_pred)) * (seq_len - 1), cum_pred
    )
    tmp["iteration"] = i
    tmp["loss"] = loss
    tmp["pred_rate"] = slope
    if train:
        print("Train  iter: %d  loss: %1.4e  pred: %.5f" % (i, loss, slope))
    else:
        print("Test   iter: %d  loss: %1.4e  pred: %.5f" % (i, loss, slope))

    return tmp
