import torch

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time
import logging
import json
import os

from torch.utils.data import Dataset
from collections import defaultdict

sns.set(font_scale=1.3)

logger = logging.getLogger(__name__)


class MapsDataset(Dataset):
    """
    Класс-датасет для задачи обучения эвристик
    """

    def __init__(self, MAP_DIR, HEURISTIC_DIR, GOAL_DIR, map2heuristic,
                 maps_size=(64, 64), heuristics_amount=50000, heuristics_per_map=10):
        self.MAP_DIR = MAP_DIR
        self.HEURISTIC_DIR = HEURISTIC_DIR
        self.GOAL_DIR = GOAL_DIR
        self.heuristics_amount = heuristics_amount
        self.map2amount = heuristics_per_map

        self.map_files = list(map2heuristic.keys())
        self.map2heuristic_files = map2heuristic
        self.maps_size = maps_size

    def __len__(self, ):
        return self.heuristics_amount

    def _get_minimal_cost(self, maps_size, goal):
        n1, n2 = maps_size
        X, Y = np.meshgrid(np.arange(n1), np.arange(n2))
        goal_numpy = goal.numpy().ravel()
        minimal_cost = torch.from_numpy(np.sqrt((X - goal_numpy[0]) ** 2 + (Y - goal_numpy[1]) ** 2)).float().unsqueeze(
            0)
        return minimal_cost

    def __getitem__(self, idx):
        FORMAT = '.npy'
        map_file = self.map_files[idx // self.map2amount]
        map = torch.tensor(np.load(os.path.join(self.MAP_DIR, map_file + FORMAT))).unsqueeze(0)

        heuristic_file = self.map2heuristic_files[map_file][idx % self.map2amount]
        heuristic = torch.tensor(np.load(os.path.join(self.HEURISTIC_DIR, map_file, heuristic_file))).unsqueeze(0)
        goal = torch.tensor(np.load(os.path.join(self.GOAL_DIR, map_file, heuristic_file))).unsqueeze(0)
        minimal_cost = self._get_minimal_cost(self.maps_size, goal)

        return map, heuristic, goal, minimal_cost


def plot_learning_curves(history, figpath, suptitle=''):
    '''
    Функция для вывода лосса и метрики во время обучения.

    :param history: (dict)
        accuracy и loss на обучении и валидации
    '''
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle(suptitle)

    plt.title('Лосс', fontsize=15)
    plt.plot(history['loss']['train'], label='train')
    plt.plot(history['loss']['val'], label='val')
    plt.ylabel('лосс', fontsize=15)
    plt.xlabel('эпоха', fontsize=15)
    plt.legend()
    plt.savefig(figpath)


def train_net(
        model,
        criterion,
        optimizer,
        train_batch_gen,
        val_batch_gen,
        device,
        num_epochs=5,
        output_dir='./results',
        desired_batch_size=16
):
    '''
    Функция для обучения модели и вывода лосса и метрики во время обучения.

    :param model: обучаемая модель
    :param criterion: функция потерь
    :param optimizer: метод оптимизации
    :param train_batch_gen: генератор батчей для обучения
    :param val_batch_gen: генератор батчей для валидации
    :param num_epochs: количество эпох

    :return: обученная модель
    :return: (dict) accuracy и loss на обучении и валидации ("история" обучения)
    '''
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    LOGFILE = os.path.join(output_dir, 'info.txt')
    history = defaultdict(lambda: defaultdict(list))
    model.to(device)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_batch_gen))
    logger.info("  Num Epochs = %d", num_epochs)

    for epoch in range(num_epochs):
        CHECKPOINT_PATH = os.path.join(output_dir, f'checkpoit_epoch_{epoch + 1}')
        train_loss = 0
        val_loss = 0
        logger.info(f'Started epoch {epoch + 1} / {num_epochs}')
        start_time = time.time()

        # Устанавливаем поведение dropout / batch_norm  в обучение
        model.train(True)
        batch_counter = 0

        for X_batch, y_batch, goal, minimal_cost in train_batch_gen:
            X_batch = torch.cat([X_batch, goal], dim=1)
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.float().to(device)
            minimal_cost = minimal_cost.float().to(device)

            pred_heuristic = model(X_batch)
            loss = criterion(pred_heuristic, y_batch, minimal_cost)
            loss.backward()
            batch_counter += len(y_batch)

            if batch_counter >= desired_batch_size:
                optimizer.step()
                optimizer.zero_grad()
                batch_counter = 0

            # Сохраняем лоссы на трейне
            train_loss += loss.detach().cpu().numpy()

        # Подсчитываем лоссы и сохраням в "историю"
        train_loss /= len(train_batch_gen)
        history['loss']['train'].append(train_loss)
        batch_counter = 0
        logger.info('Started validation')
        # Устанавливаем поведение dropout / batch_norm  в обучение
        model.train(False)
        batch_counter = 0
        for X_batch, y_batch, goal, minimal_cost in val_batch_gen:
            X_batch = torch.cat([X_batch, goal], dim=1)
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.float().to(device)
            minimal_cost = minimal_cost.float().to(device)

            pred_heuristic = model(X_batch)
            loss = criterion(pred_heuristic, y_batch, minimal_cost)

            # Сохраяняем лоссы на трейне
            val_loss += loss.detach().cpu().numpy()

        # Подсчитываем лоссы и сохраням в "историю"
        val_loss /= len(val_batch_gen)
        history['loss']['val'].append(val_loss)

        info = "Epoch {} of {} took {:.3f}s\ttraining loss (in-iteration): {:.6f}\tvalidation loss (in-iteration): {:.6f}".format(
            epoch + 1, num_epochs, time.time() - start_time, train_loss, val_loss)
        logger.info(info)

        with open(LOGFILE, 'a') as file:
            file.write(info + '\n')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, CHECKPOINT_PATH)

    FIG_PATH = os.path.join(output_dir, 'learning_curves.png')
    plot_learning_curves(history, FIG_PATH)
    with open(os.path.join(output_dir, 'traing_history.json'), 'w') as file:
        json.dump(history, file)

    return model, history
