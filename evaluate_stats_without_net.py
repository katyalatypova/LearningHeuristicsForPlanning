import argparse
import json
import numpy as np
import os
import torch

from collections import defaultdict
from torch.utils.data import random_split, DataLoader

from utils import MapsDataset
from astar import ManhattanDistance, AStar, MakePath, Map

EPS = 1e-5


def generate_starts(map, n_starts):
    x, y = np.where(map == 0)
    coords = np.array(list(zip(x, y)))
    idxs = np.random.choice(len(coords), n_starts, replace=False)
    return coords[idxs]


def get_stats_for_map_with_different_heuristics(val_gen, max_size=5000, heuristic_function=ManhattanDistance):
    stat = defaultdict(list)
    for i, (map, heuristic, goal, minimal_cost) in enumerate(val_gen):

        if i + 1 > max_size:
            print('The maximum amount of elements is seen. Breaking evaluation')
            break

        heuristic = heuristic.squeeze(0).squeeze(0).numpy()

        def ideal_heuristic(iStart, jStart, iGoal, jGoal):
            return heuristic[iStart, jStart]

        # у нас препятствия это -1, а в классе 1
        map = np.abs(map.squeeze(0).squeeze(0).numpy())
        MAP = Map()
        MAP.SetGridCells(width=64, height=64, gridCells=map)
        iGoal, jGoal = np.unravel_index(np.argmax(goal.squeeze(0).numpy()), (64, 64))
        starts = generate_starts(map, 100)

        print(f'\nStarted {i + 1} / {len(val_gen)} map\n')

        for iStart, jStart in starts:
            try:
                result = AStar(MAP, iStart, jStart, iGoal, jGoal, heuristicFunction=heuristic_function)
                ideal_result = AStar(MAP, iStart, jStart, iGoal, jGoal, heuristicFunction=ideal_heuristic)
                nodesExpanded = result[2]
                nodesOpened = result[3]
                if result[0]:
                    path = MakePath(result[1])
                    ideal_path = MakePath(ideal_result[1])
                    stat["len"].append(path[1])
                    stat["arb_delta_len"].append((path[1] - ideal_path[1]) / (ideal_path[1] + 1e-7))
                    correct = abs(path[1] - ideal_path[1]) < EPS
                    stat["corr"].append(correct)
                    print("Path found! Length: " + str(path[1]) + ". Nodes created: " + str(
                        len(nodesOpened) + len(nodesExpanded)) + ". Number of steps: " + str(
                        len(nodesExpanded)) + ". Correct: " + str(correct))
                else:
                    print("Path not found!")
                    stat["corr"].append(False)
                    stat["len"].append(0.0)
                    stat["arb_delta_len"].append(1e9)

                stat["nc"].append(len(nodesOpened) + len(nodesExpanded))
                stat["st"].append(len(nodesExpanded))

            except Exception as e:
                print("Execution error")
                print(e)
    return stat


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

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--heuristic_type", default=None, type=str, required=True,
                        help="Heuristic_type from list: [manhattan, ...]")
    parser.add_argument("--max_size", default=5000, type=int,
                        help="The maximum number of elements in the dataset that will be considered")

    args = parser.parse_args()

    MAP_DIR = args.map_dir
    HEURISTIC_DIR = args.heuristic_dir
    GOAL_DIR = args.goal_dir
    map2heuristic_path = args.map_to_heuristic
    output_dir = args.output_dir

    # TO DO
    if args.heuristic_type == 'manhattan':
        heurictic_function = ManhattanDistance
    else:
        raise NotImplementedError

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(map2heuristic_path, 'r') as file:
        map2heuristic = json.load(file)

    dataset = MapsDataset(MAP_DIR, HEURISTIC_DIR, GOAL_DIR, map2heuristic, maps_size=(64, 64))
    train_dataset, val_dataset = random_split(dataset, [40000, 10000])

    train_batch_gen = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        pin_memory=True, num_workers=2
    )
    val_batch_gen = DataLoader(
        val_dataset, batch_size=1, shuffle=True,
        pin_memory=True, num_workers=2
    )

    train_stat = get_stats_for_map_with_different_heuristics(train_batch_gen, args.max_size, heurictic_function)
    train_stat_path = os.path.join(output_dir, 'train_stat.json')
    with open(train_stat_path, 'w') as file:
        json.dump(train_stat, file)

    val_stat = get_stats_for_map_with_different_heuristics(val_batch_gen, args.max_size, heurictic_function)
    val_stat_path = os.path.join(output_dir, 'val_stat.json')
    with open(val_stat_path, 'w') as file:
        json.dump(val_stat, file)


if __name__ == "__main__":
    main()
