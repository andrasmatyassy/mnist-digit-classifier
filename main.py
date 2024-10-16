# main.py

import torch
from torch.utils.data import DataLoader
from models import MnistLinear, MnistConv, MnistResNet
from datasets import MnistDataset
from trainer import DigitClassifier
from utils import format_results, setup_logger, load_config, set_seed

MODEL_CLASSES = {
    'MnistLinear': MnistLinear,
    'MnistConv': MnistConv,
    'MnistResNet': MnistResNet
}


def main() -> None:
    set_seed()
    config = load_config()
    paths = config['paths']
    data_loader_config = config['data_loader']
    device = torch.device(
        config['device']
        if torch.cuda.is_available()
        else 'cpu'
    )
    mode = config['mode']

    # Set up loggers
    train_logger = setup_logger('train', paths['train_logs'])
    test_logger = setup_logger('test', paths['test_logs'])

    for model_config in config['models']:
        # Set up model and classifier
        model_class = MODEL_CLASSES[model_config['name']]
        model = model_class(
            dropout_rate=model_config['dropout_rate']
        ).to(device)
        classifier = DigitClassifier(
            model,
            device,
            learning_rate=model_config['learning_rate'],
        )

        model_path = f"{paths['models']}/{model.__class__.__name__}.pth"
        if mode['train']:
            # Create dataloaders
            train_data = MnistDataset(paths['train_data'])
            test_data = MnistDataset(paths['test_data'])
            train_dataloader = DataLoader(
                train_data,
                batch_size=data_loader_config['batch_size'],
                shuffle=True,
                num_workers=data_loader_config['num_workers']
            )
            test_dataloader = DataLoader(
                test_data,
                batch_size=data_loader_config['batch_size'],
                shuffle=True,
                num_workers=data_loader_config['num_workers']
            )
            # Train model on MNIST dataset
            train_logger.info(
                f"Starting training for {model.__class__.__name__}..."
            )
            classifier.train_model(
                train_dataloader,
                test_dataloader,
                epochs=model_config['epochs'],
            )
            train_logger.info(
                f"Training complete for {model.__class__.__name__}."
            )
            if mode['save_model']:
                train_logger.info(f"Saving model to {model_path}")
                classifier.save_model(model_path)

        if mode['test']:
            # Evaluate model on custom dataset
            if not mode['train']:
                test_logger.info(f"Loading model from {model_path}")
                classifier.load_model(model_path)
            custom_data = MnistDataset(paths['custom_data'], is_csv=False)
            custom_dataloader = DataLoader(
                custom_data,
                batch_size=data_loader_config['batch_size'],
                shuffle=False,
                num_workers=data_loader_config['num_workers']
            )
            digit_stats, total_correct, total_samples = (
                classifier.evaluate_dataset(custom_dataloader)
            )
            test_logger.info(format_results(
                model.__class__.__name__,
                digit_stats,
                total_correct,
                total_samples,
            ))


if __name__ == "__main__":
    main()
