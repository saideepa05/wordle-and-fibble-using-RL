# wordle-and-fibble
It is inspired by the reference repo: https://github.com/voorhs/wordle-rl. 
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
```
** Install dependencies
```bash
pip install -r requirements.txt
```
** Run the RL training
```bash
python wordle_Rlearning.py
```
### Training Setup
* The agent is trained with Q-learning using a custom environment.
* The training process involves exploration-exploitation trade-offs controlled by an epsilon decay strategy.
* The model is trained for 9 million episodes with experience replay.
* The training progress is logged using Weights & Biases (WandB). Below are the links for the wandb.
  *  Wordle: https://wandb.ai/worldunknown/world/runs/52fpv1be?workspace=user-deepanaidu0501
  *  Wordle with 1 lie (Fibble1): https://wandb.ai/worldunknown/world/runs/r4fwnt44?workspace=user-deepanaidu0501
  * Wordle with 2 lie (Fibble2): https://wandb.ai/worldunknown/world/runs/x162b2ga?workspace=user-deepanaidu0501
  * wordle with 3 lie (Fibble3): https://wandb.ai/worldunknown/world/runs/xylquk0r?workspace=user-deepanaidu0501
  * Wordle with 4 lie (Fibble4): https://wandb.ai/worldunknown/world/runs/v0zwd0bm?workspace=user-deepanaidu0501
  * Wordle with 5 lie(Fibble5): https://wandb.ai/worldunknown/world/runs/ypankmfa?workspace=user-deepanaidu0501

### Results
Test Win Rate
The following graph compares the test and train win rates of various models over time:
![W B Chart 3_21_2025, 3_35_20 PM](https://github.com/user-attachments/assets/092fc0a2-bcc5-4188-8b6f-172958a40ea6)
![W B Chart 3_21_2025, 3_35_57 PM](https://github.com/user-attachments/assets/5cf5d16a-f782-4556-8ffa-15988a226330)


