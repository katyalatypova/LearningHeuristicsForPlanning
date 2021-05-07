import argparse
import json
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from models import SmallUNet, UNet
from utils import MapsDataset

sns.set(font_scale=1.3)


def vis_heuristics(heuristic, ax, goal_pos, title=''):
    sns.heatmap(heuristic, ax=ax)
    ax.scatter(goal_pos[1], goal_pos[0], s=100, color='g')
    ax.set_title(title)


def plot_results(dataset, i, net, figpath, device):
    plt.figure(figsize=(25, 7))
    map, heuristic, goal, _ = dataset[i]
    input_ = torch.cat([map, goal], dim=0).unsqueeze(0)
    pred_heuristic = net(input_.float().to(device)).detach().cpu().numpy().squeeze(0).squeeze(0)
    goal_pos = np.unravel_index(np.argmax(np.abs(goal.squeeze(0).numpy())), (64, 64))

    # map = map.squeeze(0)
    heuristic = heuristic.squeeze(0).numpy()
    goal = goal.squeeze(0)

    ax = plt.subplot(1, 3, 1)
    vis_heuristics(heuristic, ax, goal_pos, 'Целевая эвристика')

    ax = plt.subplot(1, 3, 2)
    vis_heuristics(pred_heuristic, ax, goal_pos, 'Обученная эвристика')

    ax = plt.subplot(1, 3, 3)
    vis_heuristics(pred_heuristic - heuristic, ax, goal_pos, 'Разность')

    plt.savefig(figpath)


def main():
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--map_dir", default=None, type=str, required=True,
                        help="Folder containing maps")
    parser.add_argument("--goal_dir", default=None, type=str, required=True,
                        help="Folder containing goals for maps. See dataset class for info.")
    parser.add_argument("--heuristic_dir", default=None, type=str, required=True,
                        help="Folder containing heurisctics for maps. See dataset class for info.")
    parser.add_argument("--map_to_heuristic", default=None, type=str, required=True,
                        help="json file with maps names as keys and heuristic files as values. Note that goal and heuristic for one task should have the same names.")

    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: small, big")
    parser.add_argument("--checkpoint_path", default=None, type=str, required=True,
                        help="Checkpoint of pretrained_model.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    args = parser.parse_args()

    if args.model_type == 'small':
        model = SmallUNet()
    elif args.model_type == 'big':
        model = UNet()
    else:
        raise (ValueError, 'Model type should be in [small, big]')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dct = torch.load(args.checkpoint_path, map_location=device)['model_state_dict']
    model.load_state_dict(dct)
    model.to(device)

    MAP_DIR = args.map_dir
    HEURISTIC_DIR = args.heuristic_dir
    GOAL_DIR = args.goal_dir
    map2heuristic_path = args.map_to_heuristic
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(map2heuristic_path, 'r') as file:
        map2heuristic = json.load(file)

    dataset = MapsDataset(MAP_DIR, HEURISTIC_DIR, GOAL_DIR, map2heuristic, maps_size=(64, 64))
    indecies = np.random.randint(low=0, high=50000, size=100)

    for idx in indecies:
        map_idx = idx // 10
        heuristic_idx = idx % 10
        figpath = os.path.join(output_dir, f'map_{map_idx}_heuristic_{heuristic_idx}.png')
        plot_results(dataset, idx, model, figpath, device)


if __name__ == "__main__":
    main()
