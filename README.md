# MNIST Digit Classifier

## Description

The **mnist-digit-classifier** is a collection of neural network models—**Linear**, **Convolutional**, and **ResNet**—designed for the classic MNIST digit classification task using PyTorch. This project showcases data handling, model training, and evaluation processes.

Instead of utilizing the MNIST dataset from `torchvision.datasets`, this project employs a `.csv` file containing digits in the format `label, pixel_1, pixel_2, ... pixel_784`. These `.csv` files are sourced from [Joseph Redmon](https://pjreddie.com/projects/mnist-in-csv/).

Additionally, the project includes a custom dataset of handwritten digits in `.png` format, which I collected and annotated. I was curious to see how the models would perform on data with different characteristics compared to the original MNIST dataset. This custom dataset is normalized and formatted the same way as the original MNIST dataset to ensure uniformity.

## Installation

1. Clone the repository:
    ```bash
    git clone git@github.com:andrasmatyassy/mnist-digit-classifier.git
    cd digit_classifier
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

## Configuration

The project uses a `config.yaml` file to manage configurations. Modify this file to adjust paths, hyperparameters, and other settings.

## Running the Project

To run the project, use the following command:
```bash
python main.py
```

## Results

The evaluation of the models is saved to a `test.log` file in the `logs` directory. The evaluation is done on the custom dataset of handwritten digits.

## License

This project is licensed under the [MIT License](LICENSE).
