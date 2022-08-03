# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

import os
import time
import argparse
import numpy as np
from typing import Sequence
import gurobipy as gp

from dataset import fetch_data, DataTemplate
from eval import Evaluator
from model import LogisticRegression
from fair_fn import grad_ferm, grad_dp, loss_ferm, loss_dp
from utils import fix_seed, save2csv


def parse_args():
    parser = argparse.ArgumentParser(description='Influence Fairness')
    parser.add_argument('--dataset', type=str, default="adult", help="name of the dataset")
    parser.add_argument('--metric', type=str, default="eop", help="eop or dp")
    parser.add_argument('--seed', type=float, default=None, help="random seed")
    parser.add_argument('--alpha', type=float, default=None, help="hyperparameter in lp")
    parser.add_argument('--beta', type=float, default=None, help="hyperparameter in lp")
    parser.add_argument('--gamma', type=float, default=None, help="hyperparameter in lp")
    args = parser.parse_args()

    return args


def lp(fair_infl: Sequence, util_infl: Sequence, fair_loss: float, alpha: float, beta: float,
       gamma: float) -> np.ndarray:
    num_sample = len(fair_infl)
    max_fair = sum([v for v in fair_infl if v < 0.])
    max_util = sum([v for v in util_infl if v < 0.])

    print("Maximum fairness promotion: %.5f; Maximum utility promotion: %.5f;" % (max_fair, max_util))

    all_one = np.array([1. for _ in range(num_sample)])
    fair_infl = np.array(fair_infl)
    util_infl = np.array(util_infl)
    model = gp.Model()
    x = model.addMVar(shape=(num_sample,), lb=0, ub=1)

    if fair_loss >= -max_fair:
        print("=====> Fairness loss exceeds the maximum availability")
        model.addConstr(util_infl @ x <= 0. * max_util, name="utility")
        model.addConstr(all_one @ x <= alpha * num_sample, name="amount")
        model.setObjective(fair_infl @ x)
        model.optimize()
    else:
        model.addConstr(fair_infl @ x <= beta * -fair_loss, name="fair")
        model.addConstr(util_infl @ x <= gamma * max_util, name="util")
        model.setObjective(all_one @ x)
        model.optimize()

    print("Total removal: %.5f; Ratio: %.3f%%\n" % (sum(x.X), (sum(x.X) / num_sample) * 100))

    return 1 - x.X


def main(args):
    tik = time.time()

    if args.seed is not None:
        fix_seed(args.seed)

    """ initialization"""

    data: DataTemplate = fetch_data(args.dataset)
    model = LogisticRegression(l2_reg=data.l2_reg)
    val_evaluator, test_evaluator = Evaluator(data.s_val, "val"), Evaluator(data.s_test, "test")

    """ vanilla training """

    model.fit(data.x_train, data.y_train)
    if args.metric == "eop":
        ori_fair_loss_val = loss_ferm(model.log_loss, data.x_val, data.y_val, data.s_val)
    elif args.metric == "dp":
        pred_val, _ = model.pred(data.x_val)
        ori_fair_loss_val = loss_dp(data.x_val, data.s_val, pred_val)
    else:
        raise ValueError
    ori_util_loss_val = model.log_loss(data.x_val, data.y_val)

    """ compute the influence and solve lp """

    pred_train, _ = model.pred(data.x_train)

    train_total_grad, train_indiv_grad = model.grad(data.x_train, data.y_train)
    util_loss_total_grad, acc_loss_indiv_grad = model.grad(data.x_val, data.y_val)
    if args.metric == "eop":
        fair_loss_total_grad = grad_ferm(model.grad, data.x_val, data.y_val, data.s_val)
    elif args.metric == "dp":
        fair_loss_total_grad = grad_dp(model.grad_pred, data.x_val, data.s_val)
    else:
        raise ValueError

    hess = model.hess(data.x_train)
    util_grad_hvp = model.get_inv_hvp(hess, util_loss_total_grad)
    fair_grad_hvp = model.get_inv_hvp(hess, fair_loss_total_grad)

    util_pred_infl = train_indiv_grad.dot(util_grad_hvp)
    fair_pred_infl = train_indiv_grad.dot(fair_grad_hvp)

    sample_weight = lp(fair_pred_infl, util_pred_infl, ori_fair_loss_val, args.alpha, args.beta, args.gamma)

    """ train with weighted samples """

    model.fit(data.x_train, data.y_train, sample_weight=sample_weight)

    if args.metric == "eop":
        upd_fair_loss_val = loss_ferm(model.log_loss, data.x_val, data.y_val, data.s_val)
    elif args.metric == "dp":
        pred_val, _ = model.pred(data.x_val)
        upd_fair_loss_val = loss_dp(data.x_val, data.s_val, pred_val)
    else:
        raise ValueError
    upd_util_loss_val = model.log_loss(data.x_val, data.y_val)

    print("Fairness loss: %.5f -> %.5f; Utility loss: %.5f -> %.5f" % (
        ori_fair_loss_val, upd_fair_loss_val, ori_util_loss_val, upd_util_loss_val))

    _, pred_label_val = model.pred(data.x_val)
    _, pred_label_test = model.pred(data.x_test)

    val_res = val_evaluator(data.y_val, pred_label_val)
    test_res = test_evaluator(data.y_test, pred_label_test)

    tok = time.time()
    print("Total time: %.5fs" % (tok - tik))

    return


if __name__ == "__main__":
    args = parse_args()
    main(args)
