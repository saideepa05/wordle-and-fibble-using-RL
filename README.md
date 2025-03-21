# wordle-and-fibble
It is inspired by the reference repo: https://github.com/voorhs/wordle-rl \\
This project explores reinforcement learning (RL) applied to Wordle and Fibble, 5 letter word-guessing game. The agent learns optimal guessing strategies using deep reinforcement learning techniques.
The RL agent is trained using Q-learning and various action representations to maximize the win rate. The project compares the performance of different training configurations and learning strategies.


### Wordle is a popular word-guessing game where players try to guess a five-letter word within six attempts. After each guess, the game provides feedback on the guessed letters:
* 🟩 Green → The letter is correct and in the right position.
* 🟨 Yellow → The letter is correct but in the wrong position.
* ⬜ Gray → The letter is not in the word.

### What is Fibble?
Fibble is a variant of Wordle where one or more hints (letter colors) may be lies:
* Fibble1 → Wordle with 1 lie per game.
* Fibble2 → Wordle with 2 lies per game.
* Fibble3 → Wordle with 3 lies per game.
* Fibble4 → Wordle with 4 lies per game.
* Fibble5 → Wordle with 5 lies per game.
The more lies included, the harder it is for the RL agent to infer the correct word.

### Features
* Implementation of Wordle and Fibble environments.
* Reinforcement Learning using Q-learning.
* Replay Buffer and Prioritized Replay Buffer for experience replay.
* Various action representations (letters, words, combinations).
* Training logs and performance tracking using Weights & Biases (WandB).

### Installation
To set up and run the project, follow these steps:
** Clone the repository
```bash
git clone https://github.com/saideepa05/wordle-and-fibble.git
cd wordle-and-fibble
** Install dependencies
```bash
pip install -r requirements.txt
** Run the RL training
```bash
python wordle_Rlearning.py


