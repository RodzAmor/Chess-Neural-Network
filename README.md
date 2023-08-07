# Chess Neural Network

The Chess Neural Network is a machine learning based chess engine that uses a neural networks to evaluate positions and then uses the Minimax algorithm to perform the decision making. In Chess, the term "adoption" refers to the concept of winning against a person for 10 consecutive games. With that in mind, my primary goal for this project is to create a chess engine that is good enough to "adopt" me. I have not set an ELO goal in mind but as I test different models and optimize the program, I will likely add a ELO goal for the model and test it on Lichess bots.

This project was inspired by Sebastian Lague and his chess series as well as from my CMSC 320 Data Science class at UMD. Thank you!


## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Training the Model](#training-the-model)

## Features

- Neural network-based position evaluation.
- Minimax algorithm for decision making.
- Training on large datasets with historical chess games.
- Also trained a model based on my own Lichess games.
- Evaluation and playing against human or computer opponents to determine ELO.

## Data Source

The game data used in this project was obtained from [lichess open databases](https://database.lichess.org/). I used May 2023 in order to have over 100 million games which I split for 80% training and 20% testing. I also data obtained all my games that I played on my Lichess account (SteveMAlt) to build a model based on my playing style. However, my data set is much smaller than the open databases.

## Exploratory Data Analysis

Before building the model, I performed some exploratory data analysis on the pgn files in order to get an idea of the data and how to build a streamlined pipline for the data science process. I analyzed basic statistics on the games played as well as viewed the distribution of the player ratings. You can find more from the exploratory data analysis notebook. <!-- as well as in the pdf uploaded in my website. NOT COMPLETED YET --> 

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/rodzamor/chess-neural-network.git
cd chess-neural-network
pip install -r requirements.txt
```

## Usage

Still a WIP.


## Contributing

Contributions are welcome, feel free to clone the repository and make your own changes!
