# DL-Project-1-SP25

# Modified ResNet for CIFAR-10 Classification

This repository implements a modified ResNet architecture tailored for image classification on the CIFAR-10 dataset. The project features several key modifications and enhancements, such as configurable convolution and shortcut kernel sizes, integration of Squeeze-and-Excitation (SE) blocks, dropout regularization, and the use of a Lookahead optimizer wrapped around SGD. An emphasis is placed on maintaining a parameter budget below 5 million parameters while achieving competitive performance.

## Project Overview

- **Modified ResNet Architecture:**  
  The network builds upon the classical ResNet design. Key modifications include:
  - **BasicBlock with Dropout:** Each residual block can optionally include dropout for regularization.
  - **Configurable Kernels:** Both the main convolutional layers and the shortcut connections support configurable kernel sizes.
  - **SE Block:** A Squeeze-and-Excitation block is integrated to recalibrate feature responses.
  - **Lookahead Optimizer:** The optimizer uses a Lookahead strategy (k steps forward and 1 step back) to improve convergence.

- **Training Pipeline:**  
  The training setup includes:
  - Data augmentation (random cropping and horizontal flipping) and normalization for CIFAR-10.
  - A training loop with logging via TensorBoard.
  - Validation routines that generate metrics such as accuracy, loss curves, confusion matrix, and a classification report.
  - Checkpointing of the best performing model.
  - A scheduler (CosineAnnealingLR) to adjust the learning rate.

- **Inference on Custom Test Dataset:**  
  An inference pipeline is provided to run predictions on a custom test dataset (provided in a pickle file format) and output the results in a CSV file. The code also verifies the total number of parameters to ensure the model stays under a 5M parameter limit.

## Setup and Installation

Follow these steps to set up the project on your local machine:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/SaniyaGapchup/DL-Project-1-SP25.git
   cd DL-Project-1-SP25
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv myenv
   ```

3. **Activate the Virtual Environment:**
   - On Windows:
     ```bash
     myenv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source myenv/bin/activate
     ```

4. **Install the Required Packages:**
   Ensure that your `requirements.txt` includes packages such as:
   - torch
   - torchvision
   - numpy
   - pyyaml
   - tensorboard
   - torchsummary
   - matplotlib
   - seaborn
   - scikit-learn
   - pandas

   Then run:
   ```bash
   pip install -r requirements.txt
   ```

5. **Add the Virtual Environment to Jupyter Notebook:**
   ```bash
   pip install ipykernel
   python -m ipykernel install --user --name=myenv
   ```

6. **Install Jupyter**
   ```bash
   pip install jupyter
   ```

6. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   Open the provided notebook (e.g., `CIFAR10_DL_PROJECT_SP25.ipynb`) in your browser.

7. **Run All Cells:**
   Execute all cells in the notebook to train the model, visualize results, and run the inference script on the custom test dataset.

## Inference Script

Below is the inference script used to process a custom test dataset stored in a pickle file (`cifar_test_nolabel.pkl`):

```python
# Custom Dataset Evaluation
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np

class CustomTestDataset(Dataset):
    def __init__(self, pkl_file, transform=None):
        """
        Args:
            pkl_file (string): Path to the pkl file with structure:
            {
                b'data': numpy array of shape (N, 32, 32, 3),
                b'ids': numpy array of shape (N,)
            }
        """
        with open(pkl_file, 'rb') as f:
            self.data = pickle.load(f)
        self.images = self.data[b'data']  # Shape: (N, 32, 32, 3)
        self.ids = self.data[b'ids']      # Shape: (N,)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.uint8)
        image_id = self.ids[idx]
        if self.transform:
            image = self.transform(image)
        return image, image_id

# Define transforms for test data
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load custom test dataset
custom_testset = CustomTestDataset(
    pkl_file='cifar_test_nolabel.pkl',
    transform=transform_test
)

# Create DataLoader
custom_testloader = DataLoader(
    custom_testset,
    batch_size=100,
    shuffle=False,
    num_workers=2
)

# Set model to evaluation mode and run inference
predictions = []
ids = []
net.eval()

with torch.no_grad():
    for batch_images, batch_ids in custom_testloader:
        batch_images = batch_images.to(device)
        outputs = net(batch_images)
        _, predicted = outputs.max(1)
        predictions.extend(predicted.cpu().numpy())
        ids.extend(batch_ids.numpy())

# Save predictions to CSV
results_df = pd.DataFrame({
    'ID': ids,
    'Labels': predictions
})
results_df.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")
```

## Additional Notes

- **TensorBoard Logging:**  
  Training logs are saved to `summaries/notebook_run`. To view the logs, run:
  ```bash
  tensorboard --logdir=summaries/notebook_run
  ```

- **Model Checkpointing:**  
  The model checkpoints are automatically saved in the `checkpoints` directory whenever the validation accuracy improves.

- **Parameter Budget:**  
  A utility function is provided to count the model parameters. If the total exceeds 5 million, a warning is issued to help maintain efficiency.

## Teammates

- Saniya Gapchup, Sakshi Bhavsar, Samradnyee Shinde

## References

- **Efficient ResNets: Residual Network Design**  
  Aditya Thakur, Harish Chauhan, Nikunj Gupta 
  [arXiv:2306.12100](https://doi.org/10.48550/arXiv.2306.12100)

## Acknowledgements

- Thanks to the original ResNet authors and the PyTorch community for their invaluable contributions.
```
