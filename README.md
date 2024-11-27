# Scaling Laws Explorer

This repository contains tools and analysis for exploring language model scaling laws, particularly focusing on the relationships between compute, dataset size, and model performance as described in [Gadre et al., 2024](https://arxiv.org/abs/2403.08540).

## Key Components

- `main.py`: Implements core scaling law equations and optimization for compute/dataset relationships
- `scaling_explorer.py`: Visualization and analysis tools for model scaling behavior
- `scaling/`: Submodule from [mlfoundations/scaling](https://github.com/mlfoundations/scaling) containing experimental data and evaluation results
  - `exp_data/`: Raw experimental data including model configurations and evaluation results
  - `scaling_law_dict.pkl`: Pre-fitted scaling law parameters

## Setup

1. Clone the repository:
```bash
git clone https://github.com/kevinniechen/scalinglaws.git
cd scalinglaws
```

2. Initialize and update the scaling submodule:
```bash
git submodule add https://github.com/mlfoundations/scaling.git scaling
git submodule update --init --recursive
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

## Features

- Compute optimal dataset sizes for given compute budgets
- Analyze upstream loss and downstream error relationships
- Visualize scaling behavior across different model architectures
- Compare performance across different training datasets (C4, RedPajama, RefinedWeb)

## Usage

To explore scaling relationships:

```python
python main.py  # Analyze compute/dataset relationships
python scaling_explorer.py  # Generate scaling law visualizations
```

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Pandas
- SciPy

## Contributors

- Kevin Chen
- Justin Rose

## References

- Gadre et al. (2024). ["All Laws are Not Equal: Isolating Factors in Scaling Laws"](https://arxiv.org/abs/2403.08540)
