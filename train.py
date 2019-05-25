from model import get_model_criterion_and_optimizer, get_dataloaders, train_and_save_weights
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("num_classes", type=int)
    parser.add_argument("learning_rate", type=float)
    parser.add_argument("path_to_train", type=str)
    parser.add_argument("--path_to_val", type=str)
    args = parser.parse_args()

    model, criterion, optimizer = get_model_criterion_and_optimizer(args.model_name, 
                                                                    args.num_classes, learning_rate = args.learning_rate)


    dataloaders = get_dataloaders(args.path_to_train, validation_folder = args.path_to_val, batch_size = 128)


    train_and_save_weights(model, criterion, optimizer, dataloaders)