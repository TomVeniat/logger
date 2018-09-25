import os
import time

import visdom

import logger
import numpy as np
import torch

from builtins import range

np.random.seed(1234)


def random_data_generator():
    """ fake data generator
    """
    n_batches = np.random.randint(1, 10)
    for _ in range(n_batches):
        batch_size = np.random.randint(1, 5)
        data = np.random.normal(size=(batch_size, 3))
        target = np.random.randint(10, size=batch_size)
        yield (data, target)


def training_data():
    return random_data_generator()


def validation_data():
    return random_data_generator()


def test_data():
    return random_data_generator()


def oracle(data, target):
    """ fake metric data generator
    """
    loss = np.random.rand()
    acc1 = np.random.rand() + 70
    acck = np.random.rand() + 90

    return loss, acc1, acck

# some hyper-parameters of the experiment
lr = 0.01
n_epochs = 10

#----------------------------------------------------------
# Prepare logging
#----------------------------------------------------------

# clear env
viz = visdom.Visdom(server='http://localhost', port=8097)
viz.close(env='xp_name')


# create Experiment
xp = logger.Experiment("xp_name", use_visdom=True,
                       visdom_opts={'server': 'http://localhost', 'port': 8097},
                       time_indexing=False, xlabel='Epoch')
# log the hyperparameters of the experiment
xp.log_config({'lr': lr, 'n_epochs': n_epochs})
# create parent metric for training metrics (easier interface)
xp.ParentWrapper(tag='train', name='parent',
                 children=(xp.AvgMetric(name='loss'),
                           xp.AverageValueMeter(name='acc1', with_std=False),
                           xp.AvgMetric(name='acck'),
                           xp.ConfusionMeter(name='CM', n_classes=10)))
xp.ParentWrapper(tag='train_meter', name='parent',
                 children=(xp.AverageValueMeter(name='lossyy', with_std=True),
                           xp.AverageValueMeter(name='loss'),
                           xp.AverageValueMeter(name='acc1'),
                           xp.AverageValueMeter(name='acck')))

xp.AverageValueMeter(tag='1', name='YOYO_err', with_std=True)
xp.AverageValueMeter(tag='2', name='YOYO_err', with_std=True)
xp.AverageValueMeter(tag='3', name='YOYO_err', with_std=True)


# same for validation metrics (note all children inherit tag from parent)
xp.ParentWrapper(tag='val', name='parent',
                 children=(xp.AvgMetric(name='loss'),
                           xp.AvgMetric(name='acc1'),
                           xp.AvgMetric(name='acck'),
                           xp.ConfusionMeter('CM', n_classes=10)))

best1 = xp.BestMetric(tag="val-best", name="acc1")
bestk = xp.BestMetric(tag="val-best", name="acck")
xp.AvgMetric(tag="test", name="acc1")
xp.AvgMetric(tag="test", name="acck")

xp.plotter.set_win_opts(name="acc1", opts={'title': 'Accuracy@1'})
xp.plotter.set_win_opts(name="acck", opts={'title': 'Accuracy@k'})
xp.plotter.set_win_opts(name="loss", opts={'title': 'Loss'})

#----------------------------------------------------------
# Training
#----------------------------------------------------------

for epoch in range(n_epochs):
    # train model
    for (x, y) in training_data():
        loss, acc1, acck = oracle(x, y)
        # accumulate metrics (average over mini-batches)
        xp.Parent_Train.update(loss=loss, acc1=acc1*len(x),
                               acck=acck, n=len(x))
        xp.Parent_Train_Meter.update(lossyy=loss*len(x), loss=loss*len(x), acc1=acc1*len(x),
                               acck=acck*len(x), n=len(x))
        pred = torch.tensor(np.random.randint(10, size=y.shape))
        # pred = torch.randint(10, y.shape)
        truth = torch.tensor(y)
        xp.Cm_Train.add(pred, truth)

    # log metrics (i.e. store in xp and send to visdom) and reset
    xp.Parent_Train.log_and_reset()
    xp.Parent_Train_Meter.log_and_reset()


    for (x, y) in validation_data():
        loss, acc1, acck = oracle(x, y)
        xp.Parent_Val.update(loss=loss, acc1=acc1,
                             acck=acck, n=len(x))
        pred = torch.randint(10, y.shape)
        # pred = torch.tensor(np.random.randint(10, size=y.shape))
        truth = torch.tensor(y)
        xp.Cm_Val.add(pred, truth)
    xp.Parent_Val.log_and_reset()

    best1.update(xp.acc1_val).log()  # will update only if better than previous values
    bestk.update(xp.acck_val).log()  # will update only if better than previous values
    # time.sleep(.5)

for (x, y) in test_data():
    _, acc1, acck = oracle(x, y)
    # update metrics individually
    xp.Acc1_Test.update(acc1, n=len(x))
    xp.Acck_Test.update(acck, n=len(x))
xp.log_with_tag('test')

for t in range(25):
    for i in range(1, 4):
        for j in range(10):
            xp.get_metric('YOYO_err', str(i)).update(12+np.random.rand())

    for i in range(1, 4):
        xp.get_metric('YOYO_err', str(i)).log_and_reset()



print("=" * 50)
print("Best Performance On Validation Data:")
print("-" * 50)
print("Prec@1: \t {0:.2f}%".format(best1.value))
print("Prec@k: \t {0:.2f}%".format(bestk.value))
print("=" * 50)
print("Performance On Test Data:")
print("-" * 50)
print("Prec@1: \t {0:.2f}%".format(xp.acc1_test))
print("Prec@k: \t {0:.2f}%".format(xp.acck_test))

print("Train Confusion:")
print(xp.cm_train)
print("Validation Confusion:")
print(xp.cm_val)

#----------------------------------------------------------
# Save & load experiment
#----------------------------------------------------------

# save file
xp.to_json("my_json_log.json")  # or xp.to_pickle("my_pickle_log.pkl")

xp2 = logger.Experiment("")  # new Experiment instance
xp2.from_json("my_json_log.json")  # or xp.from_pickle("my_pickle_log.pkl")
# xp2.to_visdom(visdom_opts={'server': 'http://localhost', 'port': 8097})  # plot again data on visdom

# remove the file
os.remove("my_json_log.json")
