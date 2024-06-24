# VitLib

VitLib is a Python library that supports image processing for evaluating cell nuclei and cell membranes, as well as data augmentation and other functions to assist experiments. It provides efficient implementations of algorithms such as Narrowing With Guidance (NWG) for thinning, along with evaluation metrics for segmentation results. The library offers implementations in both Cython and pure Python for flexibility.

## Installation

You can install VitLib directly from GitHub using pip:

```bash
pip install git+https://github.com/Kotetsu0000/VitLib
```

## Features

- **NWG Thinning:** Implementation of the NWG algorithm for both symmetric and asymmetric thinning, useful for skeletonization of segmented objects.
- **Small Area Reduction:** Efficient removal of small regions in binary images based on a defined area threshold, improving segmentation quality.
- **Evaluation Metrics:** Calculation of standard nuclear area and evaluation of nuclear and membrane segmentation results using metrics like precision, recall, and F-measure.
- **Cython and Python Implementations:** Offers both Cython-optimized functions for speed and pure Python versions for better readability and debugging.

## Documentation

Detailed documentation for VitLib is available at [https://kotetsu0000.github.io/VitLib/](https://kotetsu0000.github.io/VitLib/). The documentation includes explanations of each function, parameters, usage examples, and notes on performance considerations. 

## Usage

Here are some examples of how to use VitLib:

```python
import numpy as np
from VitLib import NWG, smallAreaReduction, evaluate_nuclear_prediction

# Load your image
# ...

# Apply NWG thinning
thinned_image = NWG(binary_image)

# Remove small areas from a binary image
cleaned_image = smallAreaReduction(binary_image, area_th=50)

# Evaluate nuclear segmentation results
evaluation_metrics = evaluate_nuclear_prediction(predicted_image, ground_truth_image)
print(evaluation_metrics) 
```

# Document

[VitLib Document](https://kotetsu0000.github.io/VitLib/ "VitLib Document")

