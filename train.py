import argparse
import torch
import json
import os

from models import SmallUNet, UNet, loss
from utils import MapsDataset, train_net
from torch.utils.data import random_split, DataLoader
from multiprocessing import cpu_count


def main():
    SEED = 42
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
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    parser.add_argument('--alpha', type=float, default=0.0, required=True,
                        help="Weight for gradient loss.")
    parser.add_argument('--alpha1', type=float, default=1.0, required=True,
                        help="Weight for component of piece loss where output heuristic is less than minimal cost.")
    parser.add_argument('--alpha2', type=float, default=0.0, required=True,
                        help="Weight for component of piece loss where output heuristic is more than target cost.")

    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--desired_batch_size', type=int, default=32,
                        help="Desired batch size to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")

    args = parser.parse_args()
    alpha = args.alpha
    alpha1 = args.alpha1
    alpha2 = args.alpha2

    if args.model_type == 'small':
        model = SmallUNet()
    elif args.model_type == 'big':
        model = UNet()
    else:
        raise (ValueError, 'Model type should be in [small, big]')

    learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = lambda output, target_map, minimal_cost: loss(output, target_map, minimal_cost, device, alpha, alpha1,
                                                              alpha2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = f'alpha_{alpha}_alpha1_{alpha1}_alpha2_{alpha2}'

    MAP_DIR = args.map_dir
    HEURISTIC_DIR = args.heuristic_dir
    GOAL_DIR = args.goal_dir
    map2heuristic_path = args.map_to_heuristic
    output_dir = args.output_dir

    with open(map2heuristic_path, 'r') as file:
        map2heuristic = json.load(file)

    batch_size = args.batch_size
    num_epochs = args.num_train_epochs
    desired_batch_size = args.desired_batch_size if args.desired_batch_size > batch_size else batch_size

    config = {'learning_rate': learning_rate, 'alpha': alpha, 'alpha1': alpha1, 'alpha2': alpha2,
              'num_epochs': num_epochs, 'batch_size': batch_size, 'desired_batch_size': desired_batch_size}

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(os.path.join(output_dir, 'config.json'), 'w') as file:
        json.dump(config, file)

    dataset = MapsDataset(MAP_DIR, HEURISTIC_DIR, GOAL_DIR, map2heuristic, maps_size=(64, 64))
    train_dataset, val_dataset = random_split(dataset, [40000, 10000])
    train_batch_gen = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=cpu_count()
    )
    val_batch_gen = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=cpu_count()
    )

    _ = train_net(
        model,
        criterion,
        optimizer,
        train_batch_gen,
        val_batch_gen,
        device,
        num_epochs=num_epochs,
        output_dir=output_dir,
        desired_batch_size=desired_batch_size,
        exp_name=exp_name
    )


if __name__ == "__main__":
    main()
