# reversi0


## Introduction
A minimal working implementation of Google Deepmind's AlphaZero algorithm in Python.
Heavily based on Google Deepmind's own pseudocode from the [Supplementary Materials of their paper](https://science.sciencemag.org/content/suppl/2018/12/05/362.6419.1140.DC1?_ga=2.139898103.578007411.1578437331-922703037.1575654027).

The board representation in the game logic is taken from [this github repo](https://github.com/thomasahle/sunfish).

Runs cycles of self-play game generation and network training, intended for the CPU.

## Execution
Run training by executing

```
reversi0(Config())
```

## Disclaimer
The code haven't been tested carefully.
