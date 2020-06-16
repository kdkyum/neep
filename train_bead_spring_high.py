import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from misc.sampler import CartesianSampler
from misc.utils import logging, logging_r, save_checkpoint
from model.net import NEEP
from toy.bead_spring_high import del_medium_etpy, del_shannon_etpy, simulation, tot_entpy


def train(opt, model, optim, trajs, sampler, transform):
    model.train()
    batch, next_batch = next(sampler)

    s_prev = transform(trajs[batch].to(opt.device))
    s_next = transform(trajs[next_batch].to(opt.device))
    ent_production = model(s_prev, s_next)
    optim.zero_grad()
    loss = (-ent_production + torch.exp(-ent_production)).mean()
    loss.backward()
    optim.step()
    return loss.item()


def validate(opt, model, trajs, sampler, transform):
    model.eval()
    sampler.eval()

    ret = []
    loss = 0
    with torch.no_grad():
        for batch, next_batch in sampler:
            s_prev = transform(trajs[batch].to(opt.device))
            s_next = transform(trajs[next_batch].to(opt.device))

            ent_production = model(s_prev, s_next)
            entropy = ent_production.cpu().squeeze().numpy()
            ret.append(entropy)
            loss += (-ent_production + torch.exp(-ent_production)).sum().cpu().item()
    loss = loss / sampler.size
    ret = np.concatenate(ret)
    ret = ret.reshape(trajs.shape[0], -1)
    return ret, loss


def main(opt):
    trajs = simulation(opt.n_trj, opt.n_step, opt.n_bead, 
        opt.time_step, seed=0)
    test_trajs = simulation(opt.n_trj, opt.n_step, opt.n_bead, 
        opt.time_step, seed=3)

    mean, std = trajs.mean(axis=(0, 1)).to(opt.device), trajs.std(axis=(0, 1)).to(opt.device)
    transform = lambda x: (x - mean)/std if opt.normalize else lambda x: x

    opt.n_input = opt.n_bead
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    model = NEEP(opt)
    model = model.to(opt.device)
    optim = torch.optim.Adam(model.parameters(), opt.lr, weight_decay=opt.wd)

    train_sampler = CartesianSampler(opt.n_trj, opt.n_step, opt.batch_size)
    test_sampler = CartesianSampler(opt.n_trj, opt.n_step, opt.test_batch_size, train=False)

    ents = tot_entpy(test_trajs)

    ret_train = []
    ret_test = []

    if not os.path.exists(opt.save):
        os.makedirs(opt.save)

    for i in tqdm(range(1, opt.n_iter + 1)):
        if i % opt.record_freq == 0 or i == 1:
            preds, train_loss = validate(opt, model, trajs, test_sampler, transform)
            train_log = logging(i, train_loss, opt.time_step, preds)

            preds, test_loss = validate(opt, model, test_trajs, test_sampler, transform)
            test_log = logging_r(i, test_loss, opt.time_step, ents, preds)
            if i == 1:
                best_loss = test_loss
                best_pred_rate = test_log["pred_rate"]
            else:
                is_best = test_loss < best_loss
                if is_best:
                    best_loss = test_loss
                    best_pred_rate = test_log["pred_rate"]
                save_checkpoint(
                    {
                        "iteration": i,
                        "state_dict": model.state_dict(),
                        "best_loss": best_loss,
                        "best_pred_rate": best_pred_rate,
                        "optimizer": optim.state_dict(),
                    },
                    is_best,
                    opt.save,
                )
            test_log["best_loss"] = best_loss
            test_log["best_pred_rate"] = best_pred_rate
            ret_train.append(train_log)
            ret_test.append(test_log)
            train_sampler.train()

        train(opt, model, optim, trajs, train_sampler, transform)

    train_df = pd.DataFrame(ret_train)
    test_df = pd.DataFrame(ret_test)

    train_df.to_csv(os.path.join(opt.save, "train_log.csv"), index=False)
    test_df.to_csv(os.path.join(opt.save, "test_log.csv"), index=False)
    opt.device = "cuda" if use_cuda else "cpu"
    hparams = json.dumps(vars(opt))
    with open(os.path.join(opt.save, "hparams.json"), "w") as f:
        f.write(hparams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Neural Entropy Production Estimator for multi bead-spring model"
    )
    parser.add_argument(
        "--n-trj",
        "-M",
        type=int,
        default=1000,
        metavar="M",
        help="number of trajectories (default: 1000)",
    )
    parser.add_argument(
        "--n-step",
        "-L",
        type=int,
        default=10000,
        metavar="L",
        help="number of step for each trajectory (default: 10000)",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=1e-2,
        help="time step size of simulation (default: 0.01)",
    )
    parser.add_argument(
        "--save",
        default="./checkpoint",
        type=str,
        metavar="PATH",
        help="path to save result (default: none)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        metavar="PATH",
        help="path to load state dictionary (default: none)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        metavar="N",
        help="input batch size for training (default: 4096)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=10000,
        metavar="N",
        help="input batch size for testing (default: 10000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        metavar="LR",
        help="learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=1e-5,
        metavar="WD",
        help="weight decay (default: 1e-5)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=1000000,
        metavar="N",
        help="number of iteration to train (default: 1000000)",
    )
    parser.add_argument(
        "--record-freq",
        type=int,
        default=10000,
        metavar="N",
        help="recording frequency (default: 10000)",
    )
    parser.add_argument(
        "--n-bead",
        type=int,
        default=8,
        metavar="N",
        choices=[8, 16, 32, 64, 128],
        help="number of input neuron = number of beads (default: 8)",
    )
    parser.add_argument(
        "--n-hidden",
        type=int,
        default=256,
        metavar="N",
        help="number of hidden neuron (default: 256)",
    )
    parser.add_argument(
        "--n-layer",
        type=int,
        default=3,
        metavar="N",
        help="number of MLP layer (default: 3)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="data normalizing preprocess",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )

    opt = parser.parse_args()
    use_cuda = not opt.no_cuda and torch.cuda.is_available()
    opt.device = torch.device("cuda" if use_cuda else "cpu")

    main(opt)
