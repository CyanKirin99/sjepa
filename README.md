# S-JEPA [![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

## Introduction
S-JEPA (Spectral Joint Embedding Predictive Architecture) is a novel self-supervised learning (SSL) approach designed to learn generalizable feature representations from unlabeled spectral data by pretraining on large-scale hyperspectral datasets. It enables accurate estimation of multiple leaf traits, performing exceptionally well, especially in scenarios with scarce labeled data. Our work demonstrates the potential of SSL in extracting robust spectral features for enhanced ecosystem monitoring.

## Key Features
- **Reduced Label Dependency**: Significantly lowers the reliance on extensively labeled leaf trait data through self-supervised learning.
- **Large-Scale Pretraining**: Utilizes the largest leaf spectral dataset to date (approximately 100,000 samples) for pretraining, enhancing model generalization.
- **State-of-the-Art Performance**: Achieves cutting-edge results in predicting multiple leaf traits (e.g., an average R² of 0.784 in our study).
- **Fills Data Gaps**: Helps populate sparse trait datasets, providing more comprehensive data support for ecosystem research and insights.
- **Scalable Framework**: Establishes a scalable framework for multi-trait estimation from hyperspectral data.

## Requirements
- Python (3.12 recommended, tested with 3.12)
- PyTorch (2.5.1 recommended, tested with 2.5.1)
- Other dependencies listed in `requirements.txt`
- (Optional: CUDA version if GPU is required for optimal performance)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/CyanKirin99/sjepa.git
   cd sjepa
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Setup
1.  Download the demo dataset files (spectra and traits) from the [Releases page](https://github.com/CyanKirin99/sjepa/releases/tag/data).
2.  Create a `data/` directory in the root of the project:
    ```bash
    mkdir data
    ```
3.  Place the downloaded dataset files (e.g., `spec_demo.csv`, `trait_demo.csv`) into the `data/` directory.
4.  **Data Format**: Ensure your custom datasets align with the format of our provided demo files.
    * **Spectra File (`.csv`)**:
        The spectra file should contain a unique identifier (`uid`) for each sample, metadata about the spectral range (`start`, `end`, `length` - representing start wavelength, end wavelength, and number of bands respectively), followed by columns for each spectral band (e.g., named '0', '1', ... corresponding to wavelength indices).

        | uid   | start | end  | length | 0     | 1     | ...   |
        |-------|-------|------|--------|-------|-------|-------|
        | a0001 | 350   | 2500 | 2151   | 0.041 | 0.044 | ...   |
        | a0002 | 350   | 2500 | 2151   | 0.052 | 0.055 | ...   |
        | ...   | ...   | ...  | ...    | ...   | ...   | ...   |

    * **Traits File (`.csv`)**:
        The traits file should contain the unique identifier (`uid`) matching the spectra file, followed by columns for each leaf trait.

        | uid   | CAR   | Carbon | CHL   | LMA  | Nitrogen | ...   |
        |-------|-------|--------|-------|------|----------|-------|
        | a0001 | 0.174 | 48.2   | 0.980 | 60.6 | 3.98     | ...   |
        | a0002 | 0.180 | 47.5   | 0.995 | 62.1 | 4.05     | ...   |
        | ...   | ...   | ...    | ...   | ...  | ...      | ...   |

    The `uid` column is crucial for linking spectra to their corresponding trait values during downstream tasks.

## Usage

### 1. Pretraining with S-JEPA (Upstream Model)
1.  **Prepare Data**: Ensure your spectra dataset (e.g., `data/spec_demo.csv` or your custom dataset formatted as described above) is ready in the `data/` directory.
2.  **Run Pretraining**:
    ```bash
    python src/train/train_up.py
    ```
    This command uses the default configuration from `src/train/configs_train.yaml`. Pretrained models will be saved in the `log/log_up/` directory by default (e.g., `log/log_up/<model_size>-latest.pth.tar`).
3.  **Customized Pretraining**: To use custom settings (e.g., different model size, learning rate), you can either modify `configs_train.yaml` or use command-line arguments. For details on available arguments:
    ```bash
    python src/train/train_up.py --help
    ```


### 2. Training Decoders for Individual Traits (Downstream Models)
1.  **Prepare Data**: Your dataset should have spectra and corresponding trait values in separate CSV files, linked by a common `uid` column, and placed in the `data/` directory.
2.  **Upstream Model**:
    * Download a pretrained Upstream model from our [Releases page](https://github.com/CyanKirin99/sjepa/releases/tag/checkpoint_up) (e.g., `large_pure.pth.tar`). Place it in the expected directory (e.g., `log/log_up/large_pure.pth.tar`).
    * Alternatively, use an Upstream model you pretrained in the previous step. Ensure its path is correctly specified in `src/train/configs_train.yaml` (under `meta.up_checkpoint`) or via the `--up_checkpoint_path` command-line argument. For best results with your own model, ensure it was pretrained on a comprehensive dataset mentioned in the manuscript.
3.  **Run Downstream Training**:
    ```bash
    python src/train/train_down.py
    ```
    This uses default settings from `src/train/configs_train.yaml`, which trains a decoder for the default trait specified (e.g., `CHL`). Trained decoders are saved in `log/log_down/` by default.
4.  **Customized Downstream Training**: To train for a specific trait or use custom hyperparameters:
    ```bash
    python src/train/train_down.py --trait_name YOUR_TRAIT_NAME --up_checkpoint_path your_upstream_model.pth.tar
    ```
    For all options:
    ```bash
    python src/train/train_down.py --help
    ```

### 3. Evaluating Model Performance for a Single Trait
1.  **Prepare Data**: Use a dataset with spectra and corresponding ground-truth trait values (linked by `uid`), similar to the downstream training setup, in the `data/` directory.
2.  **Models**: Ensure you have the trained Upstream model and the specific Decoder (Downstream model) for the trait you want to evaluate. Place it in the expected directory (e.g., `log/log_down/`).
      A set of trained Downstream models is available in our [Releases page](https://github.com/CyanKirin99/sjepa/releases/tag/checkpoint_down)
3.  **Run Evaluation**:
    For example, to evaluate Chlorophyll (CHL) content:
    ```bash
    python src/eval/eval_single.py --trait_name CHL --up_checkpoint_path large_pure.pth.tar --down_checkpoint_path large_CHL.pth.tar
    ```
4.  **Customized Evaluation**: For other settings or to see all options:
    ```bash
    python src/eval/eval_single.py --help
    ```

### 4. Predicting Multiple Traits Simultaneously (Integrated Evaluation)
1.  **Prepare Data**: Have your dataset of spectra samples (e.g., `data/spec_demo.csv`) ready in the `data/` directory.
2.  **Models**: Ensure the Upstream model and all relevant Decoders (for the traits you want to predict) are available. Their paths must be correctly specified in `src/eval/configs_eval.yaml` (especially `ckpt.up_checkpoint` and `ckpt.down_checkpoint_dict`) or overridden via CLI where possible.
3.  **Run Prediction**:
    ```bash
    python src/eval/eval_integrated.py
    ```
    This will predict all traits listed in `data.trait_name_list` within `configs_eval.yaml` using the corresponding checkpoints. Results are saved as a CSV file in `log/results/` by default.
4.  **Customized Prediction**:
    ```bash
    python src/eval/eval_integrated.py --help
    ```
    For example, to predict a specific set of traits using a custom spectra file:
    ```bash
    python src/eval/eval_integrated.py --trait_name_list LMA CHL Water --spec_path data/your_custom_spectra.csv
    ```
   
## License
All code in this S-JEPA project (including our original contributions and any modifications to third-party code) is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).  
Commercial use of this project is strictly prohibited. This includes, but is not limited to, selling the software, using it in commercial products or services, or for advertising purposes.
