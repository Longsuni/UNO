import os
import random

device = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = device
import time
import torch
import torch.nn.functional as F

from utils_pack.metrics import get_MSE, get_MAE, get_MAPE
from utils_pack.utils import get_dataloader, print_model_parm_nums
from utils_pack.args import get_args
from model.UNO import UNO
import numpy as np

args = get_args()

save_path = 'experiments/incremental/{}-{}-{}'.format(
    args.model,
    args.dataset,
    args.n_channels)
torch.manual_seed(args.seed)

print("mk dir {}".format(save_path))
os.makedirs(save_path, exist_ok=True)
print('device:', device)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr[-1]


def choose_model():
    model = UNO(height=args.height, width=args.width, use_exf=args.use_exf,
                 scale_factor=args.scale_factor, channels=args.n_channels,
                 sub_region=args.sub_region,
                 scaler_X=args.scaler_X, scaler_Y=args.scaler_Y, args=args)
    return model


def load_model(task):
    load_path = '{}/best_epoch_{}.pt'.format(save_path, task)
    print("load from {}".format(load_path))
    model_state_dict = torch.load(load_path)["model_state_dict"]

    model = choose_model()
    model.load_state_dict(model_state_dict)
    if cuda:
        model = model.cuda()
        return model
    else:
        return model


criterion = F.smooth_l1_loss

total_datapath = '../datasets'
train_sequence = ["P1", "P2", "P3", "P4"]
total_mses = {"P1": [np.inf], "P2": [np.inf], "P3": [np.inf], "P4": [np.inf]}
best_epoch = {"P1": 0, "P2": 0, "P3": 0, "P4": 0}


task_id = 0

start_time = time.time()
for task in train_sequence[task_id:]:

    task_id += 1
    model = choose_model()

    if cuda:
        model = model.cuda()

    lr = args.lr

    print_model_parm_nums(model, args.model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(args.b1, args.b1))

    train_ds = get_dataloader(args,
                              datapath=total_datapath, dataset=args.dataset,
                              batch_size=args.batch_size, mode='train', task_id=task_id)
    valid_task = get_dataloader(args,
                                datapath=total_datapath, dataset=args.dataset,
                                batch_size=32, mode='valid', task_id=task_id)
    test_ds = get_dataloader(args,
                             datapath=total_datapath, dataset=args.dataset,
                             batch_size=32, mode='test', task_id=task_id)

    for epoch in range(0,args.n_epochs):
        if task_id >= 2:
            model.load_state_dict(torch.load(r"temp.pt")['model_state_dict'])

        if epoch % 20 == 0 and epoch < 50:
            lr = lr * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch >= 50:
            lr = lr * 0.8
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        epoch_start_time = time.time()

        train_loss = 0
        loss_fate = 0
        for i, (c_map, f_map, exf) in enumerate(train_ds):

            indices = random.sample(range(16 if f_map.size(0) >= 16 else 2), args.sample if f_map.size(0) >= 16 else 2)
            f_map_fate = f_map
            if task_id >= 2:

                selected_cmaps = c_map[indices]
                selected_exf = exf[indices]
                f_map_fate_indices = f_map[indices]
                results = model(selected_cmaps, selected_exf)
                loss_fate = criterion(results, f_map_fate_indices * args.scaler_Y)

            for each in indices:
                if each > 0:
                    f_map_fate[each] = f_map[each - 1]

            model.train()
            optimizer.zero_grad()
            pred_f_map = model(c_map, exf) * args.scaler_Y

            loss = criterion(pred_f_map, f_map_fate * args.scaler_Y)
            if loss_fate is not None:
                loss += loss_fate
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(c_map)
        train_loss /= len(train_ds.dataset)

        model.eval()
        val_mse, mse = 0, 0

        for j, (c_map, f_map, exf) in enumerate(valid_task):
            pred_f_map = model(c_map, exf)
            pred = pred_f_map.cpu().detach().numpy() * args.scaler_Y
            real = f_map.cpu().detach().numpy() * args.scaler_Y
            mse += get_MSE(pred=pred, real=real) * len(c_map)

        val_mse = mse / len(valid_task.dataset)

        if val_mse < np.min(total_mses[task]):
            state = {'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                     'task': task}

            best_epoch[task] = epoch
            torch.save(state, '{}/best_epoch_{}.pt'.format(save_path, task))
            torch.save(state, "temp.pt")
        total_mses[task].append(val_mse)

        log = ('Task:{}|Epoch:{}|Loss:{:.3f}|Val_MSE{:.3f}|Time_Cost:{:.2f}|Best_Epoch:{}|lr:{}\n'.format(
            task, epoch, train_loss,
            total_mses[train_sequence[task_id - 1]][-1],
            time.time() - epoch_start_time,
            best_epoch[task],
            get_learning_rate(optimizer)))
        print(log)
        f = open('{}/train_process.txt'.format(save_path), 'a')
        f.write(log)

    model = load_model(task)
    model.eval()
    total_mse, total_mae, total_mape = 0, 0, 0

    for i, (c_map, f_map, eif) in enumerate(test_ds):
        pred_f_map = model(c_map, eif)
        pred = pred_f_map.cpu().detach().numpy() * args.scaler_Y
        real = f_map.cpu().detach().numpy() * args.scaler_Y
        total_mse += get_MSE(pred=pred, real=real) * len(c_map)
        total_mae += get_MAE(pred=pred, real=real) * len(c_map)
        total_mape += get_MAPE(pred=pred, real=real) * len(c_map)

    mse = total_mse / len(test_ds.dataset)
    mae = total_mae / len(test_ds.dataset)
    mape = total_mape / len(test_ds.dataset)

    log = ('{} test: MSE={:.6f}, MAE={:.6f}, MAPE={:.6f}'.format(task, mse, mae, mape))
    f = open('{}/test_results.txt'.format(save_path), 'a')
    f.write(log + '\n')
    f.close()
    print(log)
    print('*' * 64)

log = (
    f'Total running time: {(time.time() - start_time) // 60:.0f}mins {(time.time() - start_time) % 60:.0f}s')
print(log)
f = open('{}/test_results.txt'.format(save_path), 'a')
f.write(log + '\n')
f.close()
print('*' * 64)
