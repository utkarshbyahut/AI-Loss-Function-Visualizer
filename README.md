# AI Loss Function Visualizer

An interactive PyTorch tool to visualize and compare how different loss functions affect model convergence and decision boundaries. 
Built for beginners in machine learning and structured for easy expansion into research experiments.

## Features
* Compare Binary Cross-Entropy, Mean Squared Error, and Hinge Loss
* Visualize decision boundaries for each loss function
* Plot loss curves over training epochs
* Modular, beginner‑friendly PyTorch codebase

## Project Structure
```
ai-loss-visualizer/
├── notebooks/
│   └── 01_loss_function_visualizer.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── visualize.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation
```bash
git clone https://github.com/<your-username>/ai-loss-visualizer.git
cd ai-loss-visualizer
pip install -r requirements.txt
```

## Usage
Run the beginner notebook:
```bash
jupyter notebook notebooks/01_loss_function_visualizer.ipynb
```

## Future Roadmap
* Add interactive widgets for learning rate, epochs, and batch size
* Compare optimizers (Adam, SGD, RMSprop)
* Extend to CNNs on image datasets such as CIFAR‑10
* Deploy as a web app with live training visualizations

## License
MIT License — feel free to fork, modify, and share.

> Tip: This repo is designed as a foundation project. You can fork it to create visualizers for optimizers, regularization techniques, or custom architectures.
