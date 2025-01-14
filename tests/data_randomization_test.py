import argparse
import os
import time

import torch.utils.data
from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE
from auxiliary.utils import log_experiment, log_metrics, print_val_metrics, log_time, make_deterministic
from classes.tasks.ccc.multiframe.data.TCC import TCC
from classes.tasks.ccc.multiframe.modules.conf_tccnet.ModelConfTCCNet import ModelConfTCCNet
from classes.tasks.ccc.multiframe.modules.conf_att_tccnet.ModelConfAttTCCNet import ModelConfAttTCCNet
from classes.tasks.ccc.multiframe.modules.att_tccnet.ModelAttTCCNet import ModelAttTCCNet
from classes.tasks.ccc.core.EvaluatorCCC import EvaluatorCCC
from classes.core.LossTracker import LossTracker

from vis.visualize import Visualize

# ----------------------------------------------------------------------------------------------------------------
HIDDEN_SIZE = 128
KERNEL_SIZE = 5

DATA_FOLDER = "fold_0"
EPOCHS = 2000
LEARNING_RATE = 0.00003
RANDOM_SEED = 0

PATH_TO_LOGS = os.path.join(r"./train", "tcc", "logs")

RELOAD_CHECKPOINT = False
PATH_TO_PTH_CHECKPOINT = os.path.join("trained_models", "{}_{}".format("tccnet", DATA_FOLDER), "model.pth")

MODEL_TYPE = "conf_tccnet"

MODELS = {"att_tccnet": ModelAttTCCNet, "conf_tccnet": ModelConfTCCNet, "conf_att_tccnet": ModelConfAttTCCNet}
SAL_TYPES = {"att_tccnet": "spatiotemp", "conf_tccnet": "spatiotemp", "conf_att_tccnet": "spatiotemp"}
# ----------------------------------------------------------------------------------------------------------------


def main(opt):
    model_type, hidden_size, kernel_size, deactivate = opt.model_type, opt.hidden_size, opt.kernel_size, opt.deactivate
    data_folder, epochs, learning_rate = opt.data_folder, opt.epochs, opt.lr
    reload_ckpt, path_to_ckpt = opt.reload_ckpt, opt.path_to_ckpt
    evaluator = EvaluatorCCC()

    path_to_log = os.path.join("train", "tcc", "logs", "", "{}_{}_{}".format(model_type, data_folder, + time.time()))
    path_to_destination = os.path.join("train", "tcc", "logs")
    os.makedirs(path_to_log)

    path_to_metrics_log = os.path.join(path_to_log, "metrics.csv")
    path_to_experiment_log = os.path.join(path_to_log, "experiment.json")
    log_experiment(model_type, data_folder, learning_rate, path_to_experiment_log)

    print("\nLoading data from '{}':".format(data_folder))

    training_set = TCC(train=True, data_folder=data_folder, random_labels=True)
    train_loader = DataLoader(dataset=training_set, batch_size=1, shuffle=True, num_workers=16)

    test_set = TCC(train=False, data_folder=data_folder)
    test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=16)

    training_set_size, test_set_size = len(training_set), len(test_set)
    print("Training set size: ... {}".format(training_set_size))
    print("Test set size: ....... {}\n".format(test_set_size))

    model = MODELS[model_type](hidden_size, kernel_size, SAL_TYPES[model_type])

    if reload_ckpt:
        print('\n Reloading checkpoint - pretrained model stored at: {} \n'.format(path_to_ckpt))
        model.load(path_to_ckpt)

    model.print_network()
    model.log_network(path_to_log)
    model.set_optimizer(learning_rate)

    best_val_loss, best_metrics = 100.0, evaluator.get_best_metrics()
    train_loss, val_loss = LossTracker(), LossTracker()

    for epoch in range(epochs):

        print("\n--------------------------------------------------------------")
        print("\t\t Training epoch {}/{}".format(epoch + 1, epochs))
        print("--------------------------------------------------------------\n")

        model.train_mode()
        train_loss.reset()
        start = time.time()

        for i, (x, m, y, _) in enumerate(train_loader):
            x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
            loss = model.optimize(x, y, m)
            train_loss.update(loss)

            if i % 5 == 0:
                print("[ Epoch: {}/{} - Batch: {}/{} ] | [ Train loss: {:.4f} ]"
                      .format(epoch + 1, epochs, i + 1, training_set_size, loss))

        train_time = time.time() - start
        log_time(time=train_time, time_type="train", path_to_log=path_to_experiment_log)

        val_loss.reset()
        start = time.time()

        if epoch % 5 == 0:

            print("\n--------------------------------------------------------------")
            print("\t\t Validation")
            print("--------------------------------------------------------------\n")

            with torch.no_grad():

                model.evaluation_mode()
                evaluator.reset_errors()

                for i, (x, m, y, _) in enumerate(test_loader):
                    x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
                    pred = model.predict(x, m)
                    loss = model.get_loss(pred, y).item()
                    val_loss.update(loss)
                    evaluator.add_error(loss)

                    if i % 5 == 0:
                        print("[ Epoch: {}/{} - Batch: {}/{}] | Val loss: {:.4f} ]"
                              .format(epoch + 1, EPOCHS, i + 1, test_set_size, loss))

            print("\n--------------------------------------------------------------\n")

        val_time = time.time() - start
        log_time(time=val_time, time_type="val", path_to_log=path_to_experiment_log)

        metrics = evaluator.compute_metrics()
        print("\n********************************************************************")
        print(" Train Time ... : {:.4f}".format(train_time))
        print(" Train Loss ... : {:.4f}".format(train_loss.avg))
        if val_time > 0.1:
            print("....................................................................")
            print(" Val Time ..... : {:.4f}".format(val_time))
            print(" Val Loss ..... : {:.4f}".format(val_loss.avg))
            print("....................................................................")
            print_val_metrics(metrics, best_metrics)
        print("********************************************************************\n")

        if 0 < val_loss.avg < best_val_loss:
            best_val_loss = val_loss.avg
            best_metrics = evaluator.update_best_metrics()
            print("Saving new best model... \n")
            model.save(path_to_destination)

        log_metrics(train_loss.avg, val_loss.avg, metrics, best_metrics, path_to_metrics_log)

        visualizer = Visualize(img_list=[], model_path=path_to_destination)
        visualizer.runtime()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--model_type", type=str, default=MODEL_TYPE)
    parser.add_argument('--data_folder', type=str, default=DATA_FOLDER)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--hidden_size', type=int, default=HIDDEN_SIZE)
    parser.add_argument('--kernel_size', type=int, default=KERNEL_SIZE)
    parser.add_argument('--deactivate', type=str, default=SAL_TYPES[MODEL_TYPE])
    parser.add_argument('--reload_ckpt', action="store_true")
    opt = parser.parse_args()

    opt.path_to_ckpt = os.path.join("trained_models", "no_{}".format(opt.deactivate),
                                    opt.model_type, opt.data_folder, "model.pth")

    print("\n *** Training configuration ***")
    print("\t Model type ...... : {}".format(opt.model_type))
    print("\t Data folder ..... : {}".format(opt.data_folder))
    print("\t Epochs .......... : {}".format(opt.epochs))
    print("\t Learning rate ... : {}".format(opt.lr))
    print("\t Random seed ..... : {}".format(opt.random_seed))
    print("\t Hidden size ..... : {}".format(opt.hidden_size))
    print("\t Kernel size ..... : {}".format(opt.kernel_size))
    print("\t Deactivate ...... : {}".format(opt.deactivate))

    make_deterministic(opt.random_seed)
    main(opt)
