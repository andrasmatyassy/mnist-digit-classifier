# MNIST Digit Classifier

## Description

The **mnist-digit-classifier** is a collection of neural network models—**Linear**, **Convolutional**, and **ResNet**—designed for the classic MNIST digit classification task using PyTorch. This project showcases data handling, model training, and evaluation processes.

Instead of utilizing the MNIST dataset from `torchvision.datasets`, this project employs `.csv` files containing digits in the format `label, pixel_1, pixel_2, ... pixel_784` sourced from [Joseph Redmon](https://pjreddie.com/). These `.csv` files are then made into Datasets that are passed to the DataLoaders.

Additionally, the project includes a small custom dataset of 130 digits (inputs were made with a trackpad or a stylus) drawn by 4 different people in `.png` format, which I collected and annotated. I was curious to see how the models would perform on data with different characteristics (e.g. handwriting style, digital creation) compared to the original MNIST dataset. This custom dataset is normalized and formatted the same way as the original MNIST dataset to ensure uniformity.

## Installation

1. Clone the repository:
    ```bash
    git clone git@github.com:andrasmatyassy/mnist-digit-classifier.git
    cd mnist-digit-classifier
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the MNIST dataset CSV files from Joseph Redmon and place them in the `data/mnist` directory.
    * Training set: https://pjreddie.com/media/files/mnist_train.csv
    * Test set: https://pjreddie.com/media/files/mnist_test.csv

## Configuration

The project uses a `config.yaml` file to manage configurations. Modify this file to adjust paths, hyperparameters, and other settings.

## Running the Project

To run the project, use the following command:
```bash
python main.py
```

If `mode.train` is set to `true` in the `config.yaml` file, the models will be trained and the intermediate results will be printed to the console.

## Results

Final evaluation is done on the custom dataset of handwritten digits. The evaluation of the models is saved to `logs/test.log`.

While the models perform well on the original MNIST test dataset, the performance drops significantly when evaluated on the custom dataset:

| Model | Accuracy on MNIST | Accuracy on Custom Dataset |
| --- | --- | --- |
| Linear | 0.98 | 0.63 |
| Convolutional | 0.99 | 0.72 |
| ResNet | 0.99 | 0.92 |

## License

This project is licensed under the [MIT License](LICENSE).
