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
### How lies are introduced?
Wordle with lies are introduced by randomly altering the feedback given after a player's guess. Normally, Wordle provides color-coded hints: Green (G) for correct letters in the correct position, Yellow (Y) for correct letters in the wrong position, and Gray (G) for incorrect letters. In Fibble, a set number of lies (e.g., 1 lie in Fibble1, 2 lies in Fibble2, etc.) are introduced by randomly changing non-green tiles to another incorrect value. This means a Yellow (Y) might turn into Gray (G), or a Gray (G) might turn into Yellow (Y) or even Green (G), making the feedback misleading. The game ensures randomness in selecting which tiles to alter while avoiding too many changes to Green (G) tiles to maintain fairness.
### Why Are Lies Introduced?
Lies make the game more challenging by forcing players to analyze inconsistencies and strategize beyond direct color feedback. Players must deduce which hints are false by cross-checking multiple guesses, making the game more about logical deduction than simple pattern matching. This adds an element of deception and complexity, turning Wordle into a more strategic puzzle where identifying lies is just as important as finding the correct word. 
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
  *  Wordle: https://wandb.ai/worldunknown/world?nw=nwuserdeepanaidu0501
  *  Wordle with 1 lie (Fibble1): https://wandb.ai/worldunknown/world/runs/r4fwnt44?nw=nwuserdeepanaidu0501
  * Wordle with 2 lie (Fibble2): https://wandb.ai/worldunknown/world/runs/x162b2ga?nw=nwuserdeepanaidu0501
  * wordle with 3 lie (Fibble3): https://wandb.ai/worldunknown/world/runs/xylquk0r?nw=nwuserdeepanaidu0501
  * Wordle with 4 lie (Fibble4): https://wandb.ai/worldunknown/world/runs/v0zwd0bm?nw=nwuserdeepanaidu0501
  * Wordle with 5 lie(Fibble5): https://wandb.ai/worldunknown/world/runs/ypankmfa?nw=nwuserdeepanaidu0501

* The answers.txt and guesses.txt files serve different purposes:
  * answers.txt: This file contains the list of possible secret words that the game environment selects from. In the given implementation, the environment picks one secret word from a predefined list of 2,315 words to be guessed by the agent.
  * guesses.txt: This file contains a larger vocabulary of 12,972 words that the RL model can use as possible guesses. This ensures that the agent has a broader set of words to pick from rather than being restricted to only the words that can be answers.

### Results
* Win Rate : The following graph compares the test and train win rates of wordle, fibble1, fibble2, fibble3, fibble4, fibble5 over time:
<image src="https://github.com/user-attachments/assets/092fc0a2-bcc5-4188-8b6f-172958a40ea6" width="330" height="250">
<image src="https://github.com/user-attachments/assets/5cf5d16a-f782-4556-8ffa-15988a226330"  width="330" height="250">

* The following graph compares the test and train win rates of fibble2, fibble3, fibble4, fibble5 over time:
<image src="https://github.com/user-attachments/assets/bb238fac-ddbe-421c-8b78-62b0675d5077" width="330" height="250">
<image src="https://github.com/user-attachments/assets/c1c7a4ec-dd30-4df4-a12b-422a87d01001" width="330" height="250">

### Results Summary
* Wordle (Fibble0 - No Lies): Achieved an impressive win rate of 99.8% after 50 minutes of training.
* Fibble (Wordle with Lies): The model was trained for a much longer period (9 million episodes) due to the added complexity of lies in feedback.
  * Fibble1 (1 lie per feedback) → 56.82% win rate.
  * Fibble2 (2 lies per feedback) → 0.43% win rate.
  * Fibble3 (3 lies per feedback) → 0.27% win rate.
  * Fibble4 (4 lies per feedback) → 0.16% win rate.
  * Fibble5 (5 lies per feedback) → 0.29% win rate.
The slight increase in win rate for Fibble5 compared to Fibble3 and Fibble4 is likely due to the randomization effect of lies reaching an extreme level making the misleading information less structured and therefore less deceptive.

