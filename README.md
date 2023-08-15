# Chess Neural Network

Chess engines are perhaps my most favorite application of artificial intelligence on board games. This project is my attempt to develop an AI based chess engine that is skilled enough to beat me. It consists of a chess engine that utilizes a convolutional neural network to rapidly evaluate thousands of positions along with the [minimax algorithm with alpha-beta pruning](https://www.youtube.com/watch?v=l-hh51ncgDI) to explore potential positions and determine the optimal move. In Chess, the term "adoption" refers to the concept of winning against a person for 10 consecutive games. With that in mind, my primary goal for this project is to create a chess engine that is good enough to "adopt" the average player.

Some useful papers that helped me throughout the process of developing this project is ["DeepChess: End-to-End Deep Neural Network for Automatic Learning in Chess"](https://arxiv.org/pdf/1711.09667.pdf) by Barak Oshri and Nishith Khandwala and ["Learning to Play Chess with Minimal Lookahead and Deep Value Neural Networks"](https://www.researchgate.net/publication/321028267_Learning_to_Play_Chess_with_Minimal_Lookahead_and_Deep_Value_Neural_Networks) by Matthia Sabatelli.

This project was inspired by Sebastian Lague and his chess series as well as from my CMSC 320 Data Science class at UMD. Thank you!


## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Data Source](#data-source)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Play the Chess Engine](#play-the-chess-engine)

## Features

- Neural network-based position evaluation.
- Minimax algorithm for decision making.
- Training on large datasets with historical chess games.
- Also trained a model based on my own Lichess games.
- Evaluation and playing against human or computer opponents to determine ELO.

## Data Source

The game data used in this project was obtained from [lichess open databases](https://database.lichess.org/). I used May 2023 in order to have a data set size of over 100 million games. I also included a smaller Lichess dataset of ~100 thousand games from January 2013 which is used a sample for experimenting with the data and ensuring the program works as expected before training the larger model.

I also data obtained all my games that I played on my Lichess account, [SteveMAlt](https://lichess.org/@/SteveMAlt) to build a model based on my playing style. However, my data set is much smaller than the open databases.

## Exploratory Data Analysis

Before building the model, I performed some exploratory data analysis on the pgn files in order to get an idea of the data and how to build a streamlined pipline for the data science process. I analyzed basic statistics on the games played as well as viewed the distribution of the player ratings. <!--You can find more from the exploratory data analysis notebook as well as in the pdf uploaded in my website. NOT COMPLETED YET --> 

## Data Preprocessing

According to the Lichess database website, only 6% of the pgn games have the evaluation score included. As a result, I had to both cases differently. For games with an eval included, I just used the evaluation score provided. However, for games that did not have it, I used the latest stockfish engine as of August 2023 to evaluate the position and manually add the evaluation score as a label.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/rodzamor/chess-neural-network.git
cd chess-neural-network
pip install -r requirements.txt
```

## Play the Chess Engine

You can play the Chess Neural Network with differing levels of difficulty on my website, [RodzAmor.com/chess](RodzAmor.com/chess).


## Contributing

Contributions are welcome, feel free to clone the repository and make your own changes!
