# Reference Code: https://github.com/voorhs/wordle-rl
# View Run At: https://wandb.ai/worldunknown/world?nw=nwuserdeepanaidu0501

import numpy as np
import cpprb
import torch
from typing import List, Tuple


IN_PROGRESS = 0
LOSE = 1
WIN = 2


class Wordle:
    def __init__(
        self,
        vocabulary: set,
        answers: list,
        max_guesses: int = 6,
        positive_weights=False,
        negative_weights=False,
    ):
        if not isinstance(vocabulary, set):
            raise ValueError(f'`vocabulary` arg must be a set, but given {type(vocabulary)}')
        self.vocabulary = vocabulary
        self.answers = answers
        self.n_letters = len(self.answers[0])
        
        # to generate answers randomly we sample words
        # from `self.answers` in advance and iterate through them
        self.current_answer = -1

        self.guesses_made = 0
        self.max_guesses = max_guesses

        # None, IN_PROGRESS, LOSE, WIN
        self.status = None

        # if some word was guessed [not guesses], then it
        # propability of being sampled on next iterations increases
        self.positive_weights = positive_weights
        self.negative_weights = negative_weights
        self.wins = np.zeros(len(self.answers), dtype=int)
        self.loses = np.zeros(len(self.answers), dtype=int)

    def send_guess(
        self,
        guess: str,
        output=None
    ):
        self.guesses_made += 1

        # comparing with answer
        pattern, iscorrect = self._getpattern(guess)

        # change game state
        if iscorrect:
            self.status = WIN
        elif self.guesses_made == self.max_guesses:
            self.status = LOSE

        # gameplay demonstration
        self._add_to_report(guess, pattern, output)

        return pattern

    def isover(self):
        """Whether the game is ended with result WIN or LOSE"""
        return self.status != IN_PROGRESS
    
    def iswin(self):
        if self.status == IN_PROGRESS:
            raise ValueError('Game is not ended')
        return self.status == WIN

    def _getpattern(self, guess: str):
        # initialize pattern
        pattern = [None for _ in range(self.n_letters)]

        # find green letters
        letters_left = []
        for i, a in enumerate(self.answer):
            if a == guess[i]:
                pattern[i] = 'G'
            else:
                letters_left.append(a)
        iscorrect = (len(letters_left) == 0)

        # find yellow letters
        for i, g in enumerate(guess):
            if pattern[i] is not None:   # green
                continue
            if g in letters_left:
                pattern[i] = 'Y'
                letters_left.remove(g)
            else:
                pattern[i] = 'B'

        # return pattern and flag that guess is equal to answer
        return pattern, iscorrect

    def reset_counter(self):
        self.current_answer = -1

    def reset(self, for_test=None):
        if for_test is not None:
            self.guesses_made = 0
            self.status = IN_PROGRESS
            self.answer = for_test
            self.report = f'Answer: {self.answer}\n'
            return

        # collect stats
        if self.current_answer != -1:
            ind = self.answers_indices[self.current_answer]
            self.wins[ind] += self.iswin()
            self.loses[ind] += not self.iswin()   
        
        # reset inner data
        self.guesses_made = 0
        self.status = IN_PROGRESS

        # move to next answer in list of sampled words
        self.current_answer += 1
        self.current_answer %= len(self.answers)

        if self.current_answer == 0:
            # if sampled words are over make new sample
            self.answers_indices = self._sample_answers()
        
        # update answer for new game
        ind = self.answers_indices[self.current_answer]
        self.answer = self.answers[ind]
        self.report = f'Answer {ind}: {self.answer}\n'
        
    def _sample_answers(self):
        p = Wordle._normalize(self.wins if self.positive_weights else None)
        n = Wordle._normalize(self.loses if self.negative_weights else None)

        return np.random.choice(
            len(self.answers),
            size=len(self.answers),
            p=Wordle._normalize(p+n)
        )
    
    def _normalize(weights, alpha=0.25):
        if weights == None:
            return 0
        if weights == 0:
            return None
        # small constant to prevent zero weight
        w = weights + 1e-5
        return w ** alpha / sum(w ** alpha)

    @staticmethod
    def _load_vocabulary(path, sep='\n', astype=set):
        return astype(open(path, mode='r').read().split(sep))

    def _add_to_report(self, guess, pattern, output):
        if output is None:
            return
        for g, p in zip(guess, pattern):
            letter = g.upper()
            if p == 'B':
                self.report += f'{letter:^7}'
            elif p == 'Y':
                self.report += f'{"*"+letter+"*":^7}'
            elif p == 'G':
                self.report += f"{'**'+letter+'**':^7}"

        self.report += '\n'

        # end of wordle board
        if self.isover():
            self.report += 'WIN\n\n' if self.iswin() else 'LOSE\n\n'
            open(output, mode='a+').write(self.report)

class PrioritizedReplayBuffer:
    def __init__(
        self, state_size, buffer_size=int(1e5), batch_size=64,
        alpha=0.4, beta=1, beta_growth_rate=1, update_beta_every=3000,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        n_step=1, gamma=1
    ):
        buffer_fields = {
            "state": {"shape": state_size},
            "action": {"dtype": np.int64},
            "reward": {},
            "next_state": {"shape": state_size},
            "done": {"dtype": np.bool8},
            # "indexes" and "weights" are generated automatically
        }

        self.n_step = n_step
        if n_step > 1:
            self.buffer = cpprb.PrioritizedReplayBuffer(
                buffer_size, buffer_fields, alpha=alpha,
                Nstep = {
                    # to automatically sum discounted rewards over next `n_step` steps
                    "size": n_step,
                    "gamma": gamma,
                    "rew": "reward",
                    # to automatically return `n_step`th state in `sample()`
                    "next": "next_state"
                }
            )
        else:
            self.buffer = cpprb.PrioritizedReplayBuffer(
                buffer_size, buffer_fields, alpha=alpha
            )

        self.beta = beta
        self.beta_growth_rate = beta_growth_rate
        self.update_beta_every = update_beta_every
        self.batch_size = batch_size

        # number of replays seen (but maybe not stored, because of `buffer_size` restriction)
        self.n_seen = 0
        self.device = device

        if beta_growth_rate < 1:
            raise ValueError(
                f'`beta_growth_rate` must be >=1: {beta_growth_rate}')
        if not (batch_size <= buffer_size):
            raise ValueError(
                f'Invalid sizes provided: `batch_size`={batch_size}, `buffer_size`={buffer_size}')

    def add(self, **observation):
        self.buffer.add(**observation)

        self.n_seen += 1
        if self.n_seen % self.update_beta_every == 0:
            self.beta = min(1, self.beta * self.beta_growth_rate)

    def sample(self):
        batch = self.buffer.sample(self.batch_size, self.beta)

        # send to device
        batch_device = {}
        for key, value in batch.items():
            if key == 'indexes':
                # batch['indexes'] is only used for updating priorities
                continue
            batch_device[key] = torch.from_numpy(value).to(self.device)

        return batch_device

    def get_stored_size(self):
        return self.buffer.get_stored_size()

    def update_priorities(self, indexes, tds):
        self.buffer.update_priorities(indexes, tds)

    def on_episode_end(self):
        self.buffer.on_episode_end()


class ReplayBuffer:
    def __init__(
        self, state_size, buffer_size=int(1e5), batch_size=64,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        n_step=1, gamma=1
    ):
        buffer_fields = {
            "state": {"shape": state_size},
            "action": {"dtype": np.int64},
            "reward": {},
            "next_state": {"shape": state_size},
            "done": {"dtype": np.bool8},
        }

        self.n_step = n_step
        if n_step > 1:
            self.buffer = cpprb.ReplayBuffer(
                buffer_size, buffer_fields,
                Nstep = {
                    # to automatically sum discounted rewards over next `n_step` steps
                    "size": n_step,
                    "gamma": gamma,
                    "rew": "reward",
                    # to automatically return `n_step`th state in `sample()`
                    "next": "next_state"
                }
            )
        else:
            self.buffer = cpprb.ReplayBuffer(
                buffer_size, buffer_fields
            )

        self.batch_size = batch_size
        self.device = device
        
        # number of replays seen (but maybe not stored, because of `buffer_size` restriction)
        self.n_seen = 0

        if not (batch_size <= buffer_size):
            raise ValueError(
                f'Invalid sizes provided: `batch_size`={batch_size}, `buffer_size`={buffer_size}')

    def add(self, **observation):
        self.n_seen += 1
        self.buffer.add(**observation)

    def sample(self):
        batch = self.buffer.sample(self.batch_size)

        # send to device
        batch_device = {}
        for key, value in batch.items():
            batch_device[key] = torch.from_numpy(value).to(self.device)

        return batch_device

    def get_stored_size(self):
        return self.buffer.get_stored_size()

    def on_episode_end(self):
        self.buffer.on_episode_end()

import itertools as it
import bisect
from math import ceil

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BaseAction:
    """
    Accepted sets of __init__ args:
        [vocabulary]                    init first ever action instance (before training)
        [vocabulary, nn_output]         make new action instance using fabric (agent.act)
        [vocabulary, index]             unbatch actions to single ones (__next__)
        [vocabulary, nn_output, index]  select specific actions and qfuncs (agent.learn as local value)
    
    In all cases (except the first) extra arguments can be provided
    to save some attributes in order to avoid recalculating it.
    """
    @property
    def vocabulary(self) -> List[str]:
        """Get list of all valid guesses"""
        return self._vocabulary

    @property
    def size(self):
        """Action space size."""
        return self._size

    @property
    def index(self):
        """
        If currect action instance is made from batch,
        then `index` is torch.LongTensor of shape (batch_size,)
        with indices of words in self.vocabulary corresponding
        to chosen action.

        If the instance is made from iterator,
        then `index` is a single integer.
        """
        return self._index

    @property
    def word(self) -> str:
        return self.vocabulary[self.index]

    # for compatibility with max entropy solution
    def __init__(self, word, index=0):
        self._vocabulary = [word]
        self._index = index

    @property
    def qfunc(self):
        return self._qfunc

    def _init_dict(self):
        """
        Args for __init__ to move important attributes in order to avoid recalculating it.
        """
        return {'vocabulary': self.vocabulary}
    
    def __iter__(self):
        self._iter_current = -1
        return self
    
    def __next__(self):
        """
        Used to split batched Action object to single ones.
        Copies only inheritant-specific attributes and sets the index of word.
        """
        self._iter_current += 1
        if self._iter_current < len(self.index):
            res = type(self)(
                **self._init_dict()
            )
            res.set_index(self.index[self._iter_current].cpu().item())
            return res
        raise StopIteration()
    
    def set_index(self, index):
        """For unbatching via __next__"""
        self._index = index

    def __call__(self, nn_output, **kwargs):
        """Fabric."""
        res = type(self)(
            **kwargs,
            **self._init_dict()
        )
        res.feed_nn_output(nn_output)
        return res
    
    def feed_nn_output(self, nn_output):
        """For fabric via __call__"""
        raise NotImplementedError()

    def qfunc_of_action(self, nn_output, index):
        """For retrieving qfunc for qlocal in agent.learn()"""
        raise NotImplementedError()


class ActionVocabulary(BaseAction):
    """Action is an index of word in list of possible answers."""

    def __init__(self, vocabulary: List[str]):
        if not isinstance(vocabulary, list):
            raise ValueError('`vocabulary` arg must be a list')
        self._vocabulary = vocabulary

    def feed_nn_output(self, nn_output):
        self._qfunc, self._index = nn_output.max(1, keepdim=True)
    
    def qfunc_of_action(self, nn_output, index):
        self._index = index
        self._qfunc = nn_output.gather(1, index)
        return self._qfunc

    @property
    def size(self):
        """Size of action"""
        return len(self.vocabulary)


class ActionLetters(BaseAction):
    """Action is to choose letter for each position"""

    def __init__(self, vocabulary: list, ohe_matrix, wordle_list=None):
        """
        Params
        ------
        vocabulary (List[str]): list of all valid guesses
        ohe_matrix (torch.tensor): shape (130, len(vocabulary)),
            matrix with letter-wise OHEs for each word in vocabulary
        wordle_list (Iterable[str]): full list of Wordle words
        """
        # validate input
        if not isinstance(vocabulary, list):
            raise ValueError(f'`vocabulary` arg must be a list, but given {type(vocabulary)}')
        
        self._vocabulary = vocabulary
        self.ohe_matrix = ohe_matrix
        if wordle_list is not None:
            self.ohe_matrix = ActionLetters._sub_ohe(self._vocabulary, self.ohe_matrix, wordle_list)
        self._size = self.ohe_matrix.shape[0]
   
    def _make_ohe(vocabulary, n_letters=5):
        """
        W[i, j*26+k] = vocabulary[i][j] == 65+k, i.e. indicates that
        jth letter of ith word is kth letter of alphabet
        """
        res = torch.zeros((26 * n_letters, len(vocabulary)), device=DEVICE)
            
        for i, word in enumerate(vocabulary):
            for j, c in enumerate(word):
                res[j*26+(ord(c)-97), i] = 1
        
        return res

    def _init_dict(self):
        return {
            'ohe_matrix': self.ohe_matrix,
            **super()._init_dict()
        }

    def _sub_ohe(vocabulary, ohe_matrix, wordle_list):
        """
        Select specific words from `ohe_matrix`. Used for solving subproblems.

        Params
        ------
        wordle_list, Iterable[str]: full list of Wordle words
        """
        sub_indices = []
        for word in vocabulary:
            sub_indices.append(wordle_list.index(word))
        
        return ohe_matrix[:, torch.LongTensor(sub_indices)]

    def feed_nn_output(self, nn_output):
        self._qfunc, self._index = (nn_output @ self.ohe_matrix).max(1, keepdim=True)

    def qfunc_of_action(self, nn_output, index):
        self._index = index
        self._qfunc = (nn_output @ self.ohe_matrix).gather(1, index)
        return self._qfunc
    

class ActionCombLetters(ActionLetters):
    def __init__(self, vocabulary, ohe_matrix, k, wordle_list=None):
        self.k = k
        super().__init__(vocabulary=vocabulary, ohe_matrix=ohe_matrix, wordle_list=wordle_list)
    
    def _make_ohe(vocabulary, k):
        # list of OHEs for all k
        res = []
        n_letters = len(vocabulary[0])

        for k in range(1, k+1):
            # `combs` is a list of k-lengthed tuples with indexes
            # corresponding to combination of letters in word
            combs = list(it.combinations(range(n_letters), k))
            n_combs = len(combs)

            # let's find out all unique combinations w.r.t. their positions
            # i.e. all unique pairs of (1st,3rd), (4th,5th) etc.
            unique_combs = [list() for _ in range(n_combs)]
            
            for word in vocabulary:
                for i, inds in enumerate(combs):
                    comb = ''.join([word[j] for j in inds])
                    # keep it sorted to encode quicker then
                    loc = bisect.bisect_left(unique_combs[i], comb)
                    if len(unique_combs[i]) <= loc or (unique_combs[i][loc] != comb):
                        unique_combs[i].insert(loc, comb)
            
            lens = [len(combs) for combs in unique_combs]
            
            # in worst case total_length is (5 choose k) * 26^k
            # which is 6760 for k=2
            size = sum(lens)

            # to store result
            tmp_res = torch.zeros((size, len(vocabulary)), device=DEVICE)
            
            # these barriers split OHE-vector to `n_combs` parts
            barriers = [0] + list(it.accumulate(lens))[:-1]

            for i, word in enumerate(vocabulary):
                for j, inds in enumerate(combs):
                    comb = ''.join([word[m] for m in inds])
                    
                    # `locate` is index of `comb` in `unique_combs[i]`
                    locate = barriers[j] + bisect.bisect_left(unique_combs[j], comb)
                    tmp_res[locate, i] = 1
            
            res.append(tmp_res)

        return torch.cat(res, dim=0)

    def _init_dict(self):
        return {
            'k': self.k,
            **super()._init_dict()
        }

class ActionWagons(ActionCombLetters):
    def _make_ohe(vocabulary, k):
        n_letters = len(vocabulary[0])
        n_wagons = ceil(n_letters / k)
        
        unique_wagons = [list() for _ in range(n_wagons)]

        for i, wagons in enumerate(unique_wagons):
            for word in vocabulary:
                wagon = word[i:i+k]
                loc = bisect.bisect_left(wagons, wagon)
                if len(wagons) <= loc or (wagons[loc] != wagon):
                    wagons.insert(loc, wagon)
            
        lens = [len(wagons) for wagons in unique_wagons]
        size = sum(lens)

        res = torch.zeros((size, len(vocabulary)), device=DEVICE)

        barriers = [0] + list(it.accumulate(lens))[:-1]

        for i, word in enumerate(vocabulary):
            for j, wagons in enumerate(unique_wagons):
                wagon = word[j:j+k]
                
                loc = barriers[j] + bisect.bisect_left(wagons, wagon)
                res[loc, i] = 1
        
        return res


# ======= EMBEDDING =======
from annoy import AnnoyIndex
from torch.utils.data import Dataset
import torch.nn as nn
from torch import LongTensor as LT
from torch import FloatTensor as FT
import torch as t
from typing import Literal
from math import perm
from scipy.special import comb
from tqdm.notebook import tqdm_notebook as tqdm


class WordPairsDataset(Dataset):
    """
    All pairs of Wordle words with shared letters. Each pair
    encounters as many times as many letters it shares.
    """
    def __init__(self, vocabulary, path, generate=False, int_bytes=2):
        """
        Generate (or use pregenerated) byte file with pairs of integer numbers
        representing the indexes of words in `vocabulary`.

        Params
        ------
        vocabulary, Iterable[str]: list of full Wordle words list
        path, str: path to save to or load from the dataset
        generate, bool: if False, use file `path`, if True, generate new dataset to `path`
        int_bytes, int: number of bytes to encode each integer
        """
        self.vocabulary = np.array([[ord(c) for c in word] for word in vocabulary])
        self.path = path
        self.int_bytes = int_bytes
        
        if generate:
            self._generate(self.path)

    def __len__(self):
        # if already computed
        if hasattr(self, 'size'):
            return self.size
        
        # compute len
        size = 0
        for i_letter in range(ord('a'), ord('z') + 1):
            count = np.count_nonzero(np.any(self.vocabulary == i_letter, axis=1))
            size += int(perm(count, 2))
        
        self.size = size
        return self.size

    def __getitem__(self, ind):
        src = open(self.path, 'rb')
        
        # get to ind-th pair
        src.seek(ind * 2 * self.int_bytes)
        i = int.from_bytes(src.read(self.int_bytes), byteorder='big')
        
        src.seek((ind * 2 + 1) * self.int_bytes)
        j = int.from_bytes(src.read(self.int_bytes), byteorder='big')
        
        return i, j
    
    def _generate(self, path):
        """
        Generate all pairs of words with shared letters and write to `path`.
        """
        output = open(path, 'wb')
        for i_letter in tqdm(range(ord('a'), ord('z') + 1), desc='LETTERS'):
            n_with_i_letter = np.nonzero(
                np.any(self.vocabulary == i_letter, axis=1)
            )[0].astype(np.int64)
            for i, j in tqdm(it.permutations(n_with_i_letter, 2), desc='PERMUTS'):
                output.write(int(i).to_bytes(self.int_bytes, byteorder='big'))
                output.write(int(j).to_bytes(self.int_bytes, byteorder='big'))
            

class Embedding(nn.Module):
    def __init__(self, embedding_size, vocab_size, n_negs=5):
        super().__init__()
        self.n_negs = n_negs
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.ivectors = nn.Embedding(self.vocab_size, self.embedding_size // 2)
        self.ovectors = nn.Embedding(self.vocab_size, self.embedding_size // 2)

    def forward(self, iword, owords):
        batch_size = iword.size()[0]

        nwords = iword.new_empty(batch_size, self.n_negs, dtype=torch.float).uniform_(0, self.vocab_size-1).long()
        ivectors = self.ivectors(iword).unsqueeze(2)
        ovectors = self.ovectors(owords).unsqueeze(1)
        nvectors = self.ovectors(nwords).neg()
        oloss = t.bmm(ovectors, ivectors).squeeze().sigmoid().log()
        nloss = t.bmm(nvectors, ivectors).squeeze().sigmoid().log().view(-1, self.n_negs).sum(1)
        
        return (oloss + nloss).mean().neg()
    
    def train_epoch(self, dataloader, optimizer, device):
        total_loss = 0
        
        self.train()
        for batch in tqdm(dataloader, desc='TRAIN BATCHES'):
        # for batch in dataloader:
            iwords, owords = batch[0].to(device), batch[1].to(device)
            self.zero_grad()
            loss = self.forward(iwords, owords)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        return total_loss / len(dataloader.dataset)
    
    def get_table(self):
        return torch.cat([self.ivectors.weight, self.ovectors.weight], dim=1).detach()


# in develop (critical update missing)
class ActionEmbedding(BaseAction):
    """Compatible only with ConvexQNetwork."""
    def __init__(
            self, vocabulary, emb_size, indexer:AnnoyIndex=None, nn_output=None, index=None,
            metric='euclidean', model_path='environment/embedding_model.pth'
    ):
        self._vocabulary = vocabulary
        self._size = emb_size
        
        if indexer is not None:
            self.indexer = indexer
        else:
            self.indexer = self._build_indexer(metric, model_path)

        if nn_output is not None:
            self._qfunc, self.embedding = nn_output
        if index is not None:
            self._index = index
    
    def _init_dict(self):
        return {
            'emb_size': self.size,
            'indexer': self.indexer,
            **super()._init_dict()
        }
    
    @property
    def index(self):
        """
        Finds nearest neighbour to current embedding
        among embeddings of words in `self.guesses` using AnnoyIndex.
        """ + super().__doc__
        if hasattr(self, '_index'):
            return self._index
        
        _index = []
        for emb in self.embedding.cpu():
            _index.append(
                self.indexer.get_nns_by_vector(emb, n=1)[0]
            )
        
        self._index = torch.LongTensor(_index, device=DEVICE)
        return self._index

    def act(self, index):
        """Fabric. Used in agent.act_batch()"""
        return type(self)(
            index=index,
            **self._init_dict()
        )

    def get_embeddings(self, index):
        res = []
        for i in index:
            res.append(self.indexer.get_item_vector(i))
        return torch.tensor(res)
    
    def _build_indexer(
            self, metric: Literal['angular', 'euclidean', 'manhattan', 'hamming', 'dot'],
            model_path
    ):
        # load trained nn.Module and retrieve needed matrix
        model = Embedding(self._size, len(self._vocabulary))
        model.load_state_dict(torch.load(
            model_path,
            map_location=torch.device('cpu')
        ))
        embedding_table = model.get_table()
        
        # full list of Wordle words
        wordle_words = Wordle._load_vocabulary('wordle/guesses.txt', astype=list)
        
        # resulting indexer
        res = AnnoyIndex(self._size, metric)
        
        # for each word in current problem's vocabulary
        for i, word in enumerate(self.vocabulary):
            # find location of this word in the full list
            i_emb = wordle_words.index(word)

            # add to indexer the needed vector
            res.add_item(i, embedding_table[i_emb])
        
        res.build(n_trees=10)

        return res

from collections import defaultdict
from copy import copy as PythonCopy
from typing import List

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BaseState:
    def step(self, action, pattern):
        """Update state based on agent guess and wordle pattern."""
        raise NotImplementedError()

    def reset(self):
        """Begin new episode."""
        raise NotImplementedError()

    @property
    def value(self):
        """Return vector view to put it into Q network."""
        raise NotImplementedError()

    def copy(self):
        """Return deep copy of state instance."""
        raise NotImplementedError()


class StateYesNo(BaseState):
    # inspired by https://github.com/andrewkho/wordle-solver/blob/master/deep_rl/wordle/state.py
    def __init__(self, n_letters, n_steps):
        self.n_letters = n_letters
        self.steps_left = n_steps
        self.init_steps = n_steps

        # 26 indicators that color of this letter is known
        self.isknown = np.zeros(26)

        # 26 indicators that letter is in answer
        self.isin = np.zeros(26)

        # for each 26 letters of alphabet:
        #   for each 5 letters of word:
        #       no/yes;
        self.coloring = np.zeros((26, self.n_letters, 2))

    @property
    def size(self):
        ans = 1  # number of steps left
        for arr in [self.isknown, self.isin, self.coloring]:
            ans += arr.size
        return ans

    @property
    def value(self):
        # this vector is supposed to be input of DQN network
        return np.r_[self.isknown, self.isin, self.coloring.ravel(), self.steps_left]

    def step(self, action:BaseAction, pattern, done):
        self.steps_left -= 1
        guess = action.word
        yes_letters = []

        # mark all green letters as 'yes'
        for pos, g in enumerate(guess):
            if pattern[pos] != 'G':
                continue

            let = self._getord(g)
            self.isknown[let] = 1
            self.isin[let] = 1

            # green letter strikes all the
            # alphabet letters out of this position
            self.coloring[:, pos] = [1, 0]   # 'no'
            self.coloring[let, pos] = [0, 1]  # 'yes'

            yes_letters.append(g)  # to check for duplicate black letters

        maybe_letters = []

        # for Y and B the logic is more complicated
        for pos, g in enumerate(guess):
            if pattern[pos] == 'G':  # already marked
                continue

            let = self._getord(g)
            self.isknown[let] = 1

            if pattern[pos] == 'Y':
                self.isin[let] = 1
                maybe_letters.append(g)  # to check for duplicate black letters
                self.coloring[let, pos] = [1, 0]  # 'no'

            elif pattern[pos] == 'B':
                # Wordle colors duplicate of yellow and green letters with black
                # if true number of duplicates is already met in guess
                if g in maybe_letters:
                    # this case we don't need to
                    # strike this letter out of whole word
                    # but in this position
                    self.coloring[let, pos] = [1, 0]  # 'no'
                elif g in yes_letters:
                    # this case we don't need to
                    # strike this letter out of whole word
                    # but in positions where it's not green:
                    # ('no' xor 'yes') is equivalent for ('no' is only where its not 'yes')
                    self.coloring[let, :, 0] = (
                        self.coloring[let, :, 0] != self.coloring[let, :, 1])
                else:
                    # this case we strike this letter
                    # out of whole word
                    self.coloring[let, :] = [1, 0]

    @staticmethod
    def _getord(letter):
        return ord(letter.upper()) - 65

    # start new episode
    def reset(self):
        self.isknown *= 0
        self.isin *= 0
        self.steps_left = self.init_steps
        self.coloring *= 0

    def copy(self):
        res = StateYesNo(n_letters=self.n_letters, n_steps=self.init_steps)
        res.isknown = self.isknown.copy()
        res.isin = self.isin.copy()
        res.steps_left = self.steps_left
        res.coloring = self.coloring.copy()
        return res


class StateVocabulary(StateYesNo):
    def __init__(self, answers_mask=None, answer: str = 'hello', steps=6):
        """`answers_mask` is a mask of possible answers"""
        super().__init__(answer=answer, steps=steps)
        self.init_answers_mask = answers_mask.copy()
        self.answers_mask = answers_mask.copy()

    @property
    def size(self):
        return super().size + self.answers_mask.size

    def step(self, action: BaseAction, pattern, done):
        super().step(action, pattern)
        self.answers_mask[action.value] = int(done)

    @property
    def value(self):
        return np.r_[super().value, self.answers_mask]

    def reset(self, answer):
        super().reset(answer=answer)
        self.answers_mask = self.init_answers_mask.copy()

    def copy(self):
        copy = StateVocabulary(
            answers_mask=self.answers_mask, answer=self.answer, steps=self.init_steps)
        copy.isknown = self.isknown.copy()
        copy.isin = self.isin.copy()
        copy.steps = self.init_steps
        copy.coloring = self.coloring.copy()
        return copy


class StateYesNoDordle(BaseState):
    def __init__(self, n_letters, n_boards, n_steps=None, states_list=None, freeze_list=None):
        self.n_letters = n_letters
        if n_steps is None:
            n_steps = n_boards + 5
        self.init_steps = n_steps
        self.n_boards = n_boards

        if states_list is None:
            states_list = []
            for _ in range(self.n_boards):
                states_list.append(StateYesNo(n_letters, n_steps))
        
        self.states_list: List[StateYesNo] = states_list

        # indicators of finished games
        if freeze_list is None:
            freeze_list = [False for _ in range(self.n_boards)]
        self.freeze_list = freeze_list

    @property
    def size(self):
        res = 0
        for state in self.states_list:
            res += state.size
        return res
    
    @property
    def value(self):
        res = []
        for state in self.states_list:
            res.append(state.value)
        return np.concatenate(res)

    def step(self, action, pattern_list, done_list):
        for i in range(self.n_boards):
            if self.freeze_list[i]:
                continue
            self.states_list[i].step(action, pattern_list[i])
            self.freeze_list[i] |= done_list[i]
    
    def reset(self):
        for i in range(self.n_boards):
            self.states_list[i].reset()
            self.freeze_list[i] = False
    
    def copy(self):
        states_list = []
        for state in self.states_list:
            states_list.append(state.copy())

        return StateYesNoDordle(
            n_letters=self.n_letters,
            n_steps=self.init_steps,
            n_boards=self.n_boards,
            states_list=states_list,
            freeze_list=PythonCopy(self.freeze_list)
        )

class Environment:
    def __init__(
        self, rewards: defaultdict, wordle: Wordle = None, state_instance: BaseState = None
    ):
        # supposed to be dict with keys 'B', 'Y', 'G', 'win', 'lose', 'step'
        self.rewards = rewards

        # instance of Wordle game, which we use for getting color pattern
        if wordle is None:
            wordle = Wordle()
        self.game = wordle

        # instance of envorinment state, which we use for getting input for DQN network
        self.state = state_instance
        if state_instance is None:
            self.state = StateYesNo(self.game.answer)

        # it's better to remember letters
        self.collected = {color: set() for color in ['B', 'Y', 'G']}

        # for collecting stats
        self.reward_stats = {key: 0 for key in self.rewards.keys()}

    def step(self, action: BaseAction, output):
        # convert action to str guess
        guess = action.word

        # send guess to Wordle instance
        pattern = self.game.send_guess(guess, output)

        # compute reward from pattern
        reward = self._reward(guess, pattern)

        # get to next state of environment
        self.state.step(action, pattern, self.game.isover())

        return self.state.copy(), reward, self.game.isover()

    def disable_reward_logs(self):
        self._reward = self._reward_logs_off
    
    def enable_reward_logs(self):
        self._reward = self._reward_logs_on

    def _reward_logs_on(self, guess, pattern):
        # to save result
        result = 0

        def assign_reward(key):
            rew = self.rewards[key] 
            self.reward_stats[key] += abs(rew)
            return rew

        # reward (supposed to be negative) for any guess
        result += assign_reward('step')

        # reward for each letter
        for i, color in enumerate(pattern):
            if guess[i] not in self.collected[color]:
                result += assign_reward(color)
                self.collected[color].add(guess[i])
            elif 'repeat' in self.rewards.keys() and color == 'G':
                result += assign_reward('repeat')
        
        # if end of episode
        if self.game.isover():
            result += assign_reward('win' if self.game.iswin() else 'lose')
        
        return result

    def _reward_logs_off(self, guess, pattern):
        # reward (supposed to be negative) for any guess
        result = self.rewards['step']

        # reward for each letter
        for i, color in enumerate(pattern):
            if guess[i] not in self.collected[color]:
                result += self.rewards[color]
                self.collected[color].add(guess[i])
            elif 'repeat' in self.rewards.keys() and color == 'G':
                result += self.rewards['repeat']
        
        # if end of episode
        if self.game.isover():
            result += self.rewards['win'] if self.game.iswin() else self.rewards['lose']
        
        return result

    def reset(self, for_test=None):
        self.game.reset(for_test)
        self.state.reset()
        self.collected = {color: set() for color in ['B', 'Y', 'G']}
        
        return self.state.copy()
    
    def get_test_size(self):
        return len(self.game.answers)

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from collections import defaultdict
from functools import partial
from typing import Literal
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(
            self, state_size, action_size, seed=None, hidden_size=256,
            n_hidden_layers=1, skip=True, combine_method: Literal['sum', 'concat'] = 'sum', **kwargs):
        super().__init__()
        if seed is not None:
            self.seed = torch.manual_seed(seed)
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.skip = skip
        self.combine_method = combine_method
        
        layers = [nn.Linear(state_size, hidden_size)]
        for _ in range(n_hidden_layers):
            layers.append(nn.LazyLinear(hidden_size))
        
        self.layers = nn.ModuleList(layers)
        self.output = nn.LazyLinear(action_size)

        forward = self._forward_mlp
        if self.skip:
            if self.combine_method == 'sum':
                forward = self._forward_skip_sum
            elif self.combine_method == 'concat':
                forward = self._forward_skip_concat
        
        self._forward = forward

    def forward(self, x):
        return self._forward(x)

    def _forward_mlp(self, x):
        """Default multilayer perceptron."""
        for fc in self.layers:
            x = F.relu(fc(x))
        
        return self.output(x)
    
    def _forward_skip_sum(self, x):
        """ResNet-like skip-connections."""
        # first two activations
        y = F.relu(self.layers[0](x))
        z = F.relu(self.layers[1](y))
        x = [y, z]

        # the rest of activations
        for fc in self.layers[2:]:
            x.append(F.relu(fc(x[-2] + x[-1])))
        
        return self.output(x[-2] + x[-1])

    def _forward_skip_concat(self, x):
        """DenseNet-like skip-connections."""
        prev_features = [x]

        for fc in self.layers[1:]:
            cur_features = torch.cat(prev_features, dim=1) 
            x = F.relu(fc(cur_features))
            prev_features.append(x)

        cur_features = torch.cat(prev_features, dim=1) 
        return self.output(cur_features)
    
    def fine_tune(self, layers_ind=None):
        if layers_ind is None:
            layers_ind = range(self.n_hidden_layers+1)
        for i in layers_ind:
            self.layers[i].requires_grad_(False)

    def load_backbone(self, path, layers_ind=None):
        backbone = QNetwork(
            self.state_size, self.action_size, self.seed,
            self.hidden_size, self.n_hidden_layers, self.skip
        ).float().to(DEVICE)

        backbone.load_state_dict(torch.load(path, map_location=DEVICE))

        if layers_ind is None:
            layers_ind = range(self.n_hidden_layers+1)

        for i in layers_ind:
            self.layers[i].load_state_dict(backbone.layers[i].state_dict())


class OldQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=0, **kwargs):
        super().__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        y = F.relu(self.fc2(x))

        return self.fc3(x+y)

# ======= CONVEX SOLUTION (FAILED YET) ======= 


class ConvexQNetwork(nn.Module):
    class _ParametrizePositive(nn.Module):
        def forward(self, weights):
            return F.softplus(weights)
    
    def __init__(self, state_size, emb_size, hidden_size, optim_steps) -> None:
        super().__init__()

        self.state_size = state_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.optim_steps = optim_steps

        self.Ws = nn.Linear(self.state_size, self.hidden_size)
        self.Wu_1 = nn.Linear(self.state_size, self.hidden_size) 
        self.Wz_1 = nn.Linear(self.emb_size, self.hidden_size)
        
        self.Wz_2 = nn.Linear(self.hidden_size, 1)
        self.Wu_2 = nn.Linear(self.hidden_size, 1)
        self.Wa = nn.Linear(self.emb_size, 1)

        P.register_parametrization(self.Wz_1, 'weight', self._ParametrizePositive())
        P.register_parametrization(self.Wz_2, 'weight', self._ParametrizePositive())

    def forward(self, s, a=None):
        # forward pass through network
        def minus_Q(s, a):
            u1 = F.relu(self.Ws(s))
            z1 = F.relu(self.Wz_1(a) + self.Wu_1(s))
            z2 = self.Wu_2(u1) + self.Wz_2(z1) + self.Wa(a)
            return z2
        
        if a is None:
            # solve optimization problem, so gradients wrt
            # net params are unneccessary to be computed
            for param in self.parameters():
                param.requires_grad = False

            # L-BFGS optimization
            a, self.conv = Optimizer(minus_Q, s, self.optim_steps, self.emb_size).solve()

            # enable gradients back
            for param in self.parameters():
                param.requires_grad = True
        
        # return Q(s,a), a
        return minus_Q(s, a) * (-1), a


class Optimizer:
    def __init__(self, obj, s, optim_steps, a_size):
        self.obj = obj
        self.s = s
        self.optim_steps = optim_steps
        self.a_size = a_size
    
    def _closure(self, s, a, conv, optimizer):
        """
        Argument for torch.optim.LBFGS.step
        """
        # forward pass to compute objective
        obj = self.obj(s, a)
        conv.append(obj.item())

        # delete previous gradients wrt self.a[closure.ind]
        optimizer.zero_grad()
        
        # compute gradients wrt self.a
        obj.backward()

        return obj

    def solve(self):
        """L-BFGS optimization."""
        res = []

        # for each state in batch
        conv = []
        for s in self.s:
            # initial approximation for action embedding
            a = nn.Parameter(torch.randn(self.a_size))

            # L-BFGS optimizer for one action, because torch computes
            # gradients for scalar only
            optimizer = torch.optim.LBFGS([a], max_iter=self.optim_steps)
            
            # to store convergence history for current batch elem
            conv.append([])
            
            # start optimization algorithm
            closure = partial(self._closure, s=s, a=a, conv=conv[-1], optimizer=optimizer)
            optimizer.step(closure)

            # collect resulting action
            res.append(a.detach())
        
        return torch.stack(res), conv

import random
from functools import partial
from typing import List, Union
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(
        self, state_size, action_instance:BaseAction, replay_buffer,
        gamma=1, tau=1e-3, optimize_interval=8,
        agent_path=None, **model_params
    ):
        self.state_size = state_size
        self.action = action_instance
        self.memory = replay_buffer

        # Q-Network
        self.define_model(**model_params)
        
        # load checkpoint
        if agent_path is not None:
            if 'local' in agent_path.keys():
                self.qnetwork_local.load_state_dict(torch.load(agent_path['local']))
            if 'target' in agent_path.keys():
                self.qnetwork_target.load_state_dict(torch.load(agent_path['target']))
            if 'buffer' in agent_path.keys():
                self.memory.buffer.load_transitions(agent_path['buffer'])
        
        if agent_path is not None and 'optimizer' in agent_path.keys():
            self.optimizer.load_state_dict(torch.load(agent_path['optimizer']))
        
        self.criterion = nn.MSELoss()
        self.loss = None

        # to implement pytorch-like logic of network regimes train and eval
        # if True, then `learn()` method won't be called
        self.eval = True
        self.eps = None

        # Initialize time steps
        self.t = 0

        # update params
        self.gamma = gamma
        self.tau = tau
        self.optimize_interval = optimize_interval

        # validation
        if not (0 <= gamma <= 1):
            raise ValueError(
                f'Discount factor `gamma` must be float in [0,1], but given: {gamma}')
        if not (0 <= tau <= 1):
            raise ValueError(
                f'Soft update coefficient `tau` must be float in [0,1], but given: {tau}')

    def define_model(self, **model_params):
        self.device = model_params.get('device', torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        model = model_params.get('model', QNetwork)
        self.qnetwork_local = model(self.state_size, self.action.size, **model_params).float().to(self.device)
        self.qnetwork_target = model(self.state_size, self.action.size, **model_params).float().to(self.device)
        self.optimizer = model_params.get('optimizer', partial(Adam, lr=5e-4))(self.qnetwork_local.parameters())

    def add(self, state: Union[np.ndarray, BaseState], action: BaseAction, reward, next_state: BaseState, done):
        """
        Runs internal processes:
        - update replay buffer state
        - run one epoch train
        """
        # save experience in replay memory
        self.memory.add(
            state=state if isinstance(state, np.ndarray) else state.value,
            action=action.index,
            reward=reward,
            next_state=next_state.value,
            done=done)
        if done:
            self.memory.on_episode_end()

        # check for updates
        self.t += 1

        if self.t % self.optimize_interval == 0 and not self.eval:
            self.learn()

    def act_single(self, state: BaseState):
        """Returns action for given state"""
        if isinstance(self.action, ActionEmbedding):
            return self.act_single_embedding(state)
        
        nn_output = None
        if random.random() > self.eps:
            # greedy action based on Q function
            self.qnetwork_local.eval()
            with torch.no_grad():
                nn_output = self.qnetwork_local(
                    torch.from_numpy(state.value).float().unsqueeze(0).to(self.device)
                )
        else:
            # exploration action
            nn_output = torch.randn(1, self.action.size)

        return self.action(nn_output)
    
    def act_batch(self, states: List[np.ndarray]):
        """Returns actions for given batch of states"""
        play_batch_size = len(states)
        states = np.array(states)
        
        # for each game define what action to choose: greedy or exporative
        explore_ind = []
        greedy_ind = []
        events = np.random.uniform(low=0, high=1, size=play_batch_size)
        for i in range(play_batch_size):
            if events[i] < self.eps:
                explore_ind.append(i)
            else:
                greedy_ind.append(i)

        if isinstance(self.action, ActionEmbedding):
            return self.act_batch_embedding(states, explore_ind, greedy_ind)

        # to store result
        nn_output = torch.empty((play_batch_size, self.action.size)).to(self.device)
        
        # greedy action
        self.qnetwork_local.eval()
        with torch.no_grad():
            nn_output[greedy_ind] = self.qnetwork_local(
                torch.from_numpy(states[greedy_ind])
                    .float()
                    .to(self.device)
            )
        
        # explorative action
        nn_output[explore_ind] = torch.randn(len(explore_ind), self.action.size).to(self.device)

        # convert to `BaseAction` inheritant
        return self.action(nn_output)

    def act_single_embedding(self, state):
        index = None
        if random.random() > self.eps:
            # greedy action based on Q function
            self.qnetwork_local.eval()
            with torch.no_grad():
                greedy_act = self.qnetwork_local(
                    torch.from_numpy(state.value).float().unsqueeze(0).to(self.device)
                )
            index = self.action(greedy_act).index
        else:
            # exploration action
            index = torch.tensor(
                np.random.choice(len(self.action.vocabulary))
            )

        return self.action.act(index=index)

    def act_batch_embedding(self, states: np.ndarray, explore_ind, greedy_ind):
        """Ugly branch to incorporate ActionEmbedding cases."""
        
        # indices of words in vocabulary
        index = torch.empty(states.shape[0]).long()

        # make greedy actions
        if greedy_ind:
            self.qnetwork_local.eval()
            with torch.no_grad():
                # tuple (qfunc, embeddings)
                greedy_acts = self.qnetwork_local(
                    torch.from_numpy(states[greedy_ind])
                        .float()
                        .to(self.device)
                )
            # calculate indexes (knn)
            index[greedy_ind] = self.action(greedy_acts).index

        if explore_ind:
            # calculate explorative indexes
            index[explore_ind] = torch.from_numpy(
                np.random.choice(len(self.action.vocabulary), len(explore_ind))
            )
        
        # convert to ActionEmbedding object
        return self.action.act(index=index)

    def learn(self):
        """Update net params using batch sampled from replay buffer"""
        batch = self.memory.sample()

        # Q-function
        q_target = None
        self.qnetwork_target.eval()
        if not isinstance(self.action, ActionEmbedding):
            nn_output = self.qnetwork_target(batch['next_state']).detach()
            q_target = self.action(nn_output).qfunc
        else:
            q_target, _ = self.qnetwork_target(batch['next_state'])

        # discounted return
        expected_values = batch['reward'] + self.gamma ** self.memory.n_step * q_target * (~batch['done'])

        # predicted return
        q_local = None
        self.qnetwork_local.train()
        if not isinstance(self.action, ActionEmbedding):
            nn_output = self.qnetwork_local(batch['state'])
            q_local = self.action.qfunc_of_action(nn_output, index=batch['action'].long())
        else:
            embeddings = self.action.get_embeddings(batch['action'].long())
            q_local, _ = self.qnetwork_local(batch['state'], embeddings)

        # MSE( Q_L(s_t, a_t); r_t + gamma * max_a Q_T(s_{t+1}, a) )
        if 'weights' in batch.keys():
            loss = torch.sum(batch['weights'] * (q_local - expected_values) ** 2)
        else:
            loss = torch.sum((q_local - expected_values) ** 2)

        # to print during training
        self.loss = torch.sqrt(loss).cpu().item()

        # train local network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        if 'indexes' in batch.keys():
            # update priorities basing on TD-error
            tds = (expected_values - q_local.detach()).abs().cpu().numpy()
            self.memory.update_priorities(batch['indexes'], tds)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        ¸_target = Ä * ¸_local + (1 - Ä) * ¸_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data)

    def dump(self, nickname, t):
        agent_path = {
            'local': f'{nickname}/local-{t}.pth',
            'target': f'{nickname}/target-{t}.pth',
            'buffer': f'{nickname}/buffer-{t}.npz',
            'optimizer': f'{nickname}/optimizer-{t}.pth',
        }
        torch.save(self.qnetwork_local.state_dict(), agent_path['local'])
        torch.save(self.qnetwork_target.state_dict(), agent_path['target'])
        self.memory.buffer.save_transitions(agent_path['buffer'])
        torch.save(self.optimizer.state_dict(), agent_path['optimizer'])

        return agent_path
    
    def load_backbone(self, model_path):
        self.qnetwork_local.load_backbone(model_path)
        self.qnetwork_target.load_backbone(model_path)

    def fine_tune(self):
        self.qnetwork_local.fine_tune()
        self.qnetwork_target.fine_tune()

import wandb
wandb.login(key='Enter your wandb API key') #Enter your API Key

import pandas as pd
from tqdm.notebook import tqdm
from collections import deque, defaultdict
from functools import partial
from time import time
from datetime import datetime
import os
from typing import List, Union, Literal

class RLFramework:
    """Basic utilities with agent, environment and game in one framework for experiments with training and testing."""

    def __init__(
            self, agent: Agent,
            train_env: List[Environment],
            test_word_list: List[str],
            nickname,
            detailed_logging=False,
            logging_interval=None,
            n_episodes=80000,
            n_episodes_warm=80,
            play_batch_size=8,
            eps_start=1.0, eps_end=0.05, eps_decay=0.999
        ):
        """
        Params
        ------
            logging_interval    (int): defines size of sliding window of batches to calculate and print stats from
            checkpoint_interval (int): defines the interval for saving agent's net params
            n_batches           (int): number of batches to play and learn from during training
            n_batches_warm      (int): size of initial experience to be collected before training
            play_batch_size     (int): to strike a balance between the amount of incoming replays and size of the training batch
            is_parallel        (bool): if True, then self.play_batch_parallel() is used, self.play_batch_successively() otherwise
            nickname            (str): directory name for saving checkpoints and other files
        """
        self.agent = agent

        self.train_env_list = train_env
        self.test_word_list = test_word_list

        self.n_episodes = n_episodes
        self.n_episodes_warm = n_episodes_warm
        self.play_batch_size = play_batch_size

        self.detailed_logging = detailed_logging
        if logging_interval is None:
            logging_interval = max(n_episodes // 8, 1)
        self.logging_interval = logging_interval
        
        # directory for txt, pth, npz
        i = 1
        path = nickname
        while os.path.exists(path):
            path = nickname + f' ({i})'
            i += 1
        nickname = path
        os.mkdir(nickname)

        self.nickname = nickname

        self.train_output = None

        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end

    def train(self):
        # duplicate std output to file
        self.logs_output = f'{self.nickname}/logs.txt'
        self.n_episodes_played = 0
        self.elapsed_time = 0

        # ======= COLLECT INITIAL EXPERIENCE =======

        # don't update net and 100% explore
        self.agent.eps = 1
        self.agent.eval = True
        pbar = tqdm(total=self.n_episodes_warm, desc='WARM EPISODES')
        self.play_train(self.n_episodes_warm, pbar=pbar)
        pbar.close()

        # ======= TRAINING =======

        # firstly, test initial agent
        self.agent.eval = True
        self.test_initial()

        # slowly decreasing exporation
        self.agent.eps = self.eps_start

        # for i_batch in tqdm(range(1, self.n_batches+1), desc='TRAIN BATCHES'):
        pbar = tqdm(total=self.n_episodes, desc='TRAIN EPISODES')
        for i_epoch in range(self.n_episodes//self.logging_interval):
            test_output = f'{self.nickname}/test-{i_epoch}.txt'
            train_output = f'{self.nickname}/train-{i_epoch}.txt'

            self.agent.eval = False
            train_stats = self.play_train(epoch_size=self.logging_interval, pbar=pbar, output=train_output)
            
            self.agent.eval = True
            test_stats = self.play_test(output=test_output)
            
            self.save_checkpoint(i_epoch)
            self.log(train_stats, test_stats, game_output=test_output)
        pbar.close()

    def save_checkpoint(self, i):
        self.agent.dump(self.nickname, i) 

    def play_batch_parallel(self, output):
        envs_number = len(self.train_env_list)
        
        state_size = self.train_env_list[0].state.size
        states = np.empty((envs_number, state_size))

        # reset all environments
        for env in self.train_env_list:
            env.reset()
            env.disable_reward_logs()

        # to store stats
        batch_scores = np.zeros(envs_number)
        batch_wins = np.zeros(envs_number, dtype=bool)
        
        all_is_over = False
        while not all_is_over:
            # collect batch of states from envs that are not finished yet
            indexes = []
            for i, env in enumerate(self.train_env_list):
                if env.game.isover():
                    continue

                states[i] = env.state.value
                indexes.append(i)
            
            # feed batch to agent
            actions = self.agent.act_batch(states[indexes])

            all_is_over = True
            for i, action in zip(indexes, actions):
                # send action to env
                next_state, reward, done = self.train_env_list[i].step(action, output)

                # save replay to agent's buffer
                self.agent.add(states[i], action, reward, next_state, done)
            
                # collect stats
                batch_scores[i] += reward

                if done:
                    batch_wins[i] = self.train_env_list[i].game.iswin()
                else:
                    all_is_over = False
        
        return batch_scores, batch_wins

    def play_train(self, epoch_size, pbar=None, output=None):
        n_batches = np.ceil(epoch_size / self.play_batch_size).astype(int)
        if n_batches == 0:
            return
        
        scores = []
        wins = []
        start = time()
        
        for _ in range(n_batches):
            batch_scores, batch_wins = self.play_batch_parallel(output)
            
            if pbar is not None:
                pbar.update(self.play_batch_size)
            
            scores.append(batch_scores)
            wins.append(batch_wins)

            # decrease exploration chance
            self.agent.eps = max(self.eps_end, self.eps_decay * self.agent.eps)

        n_episodes_played = n_batches * self.play_batch_size
        scores = np.concatenate(scores)
        wins = np.concatenate(wins)
        elapsed_time = time() - start

        return n_episodes_played, scores, wins, elapsed_time

    def play_test(self, output=None):
        # test agent without exploration
        eps = self.agent.eps
        self.agent.eps = 0

        # list of game states
        envs_number = len(self.train_env_list)
        state_size = self.train_env_list[0].state.size
        states = np.empty((envs_number, state_size))

        # to store stats
        scores = []
        wins = []
        steps = []

        # reset all environments
        n_words_guessed = 0
        for env in self.train_env_list:
            env.reset(self.test_word_list[n_words_guessed])
            env.enable_reward_logs()
            n_words_guessed += 1
            env.score = 0
            env.steps = 0
        
        epoch_size = len(self.test_word_list)
        i_word = 0
        while i_word < epoch_size:
            indexes = []
            # collect batch of states from envs
            for i, env in enumerate(self.train_env_list):
                if env.game.isover():
                    # start new game
                    if n_words_guessed == epoch_size:
                        continue

                    env.reset(self.test_word_list[n_words_guessed])
                    n_words_guessed += 1
                    env.score = 0
                    env.steps = 0

                states[i] = env.state.value
                indexes.append(i)
            
            # feed batch to agent
            actions = self.agent.act_batch(states[indexes])

            for i, action in zip(indexes, actions):
                env = self.train_env_list[i]

                # send action to env
                next_state, reward, done = env.step(action, output)
                
                # collect stats
                env.score += reward
                env.steps += 1

                # save replay to agent's buffer
                self.agent.add(states[i], action, reward, next_state, done)

                if done:
                    # collect stats
                    scores.append(env.score)
                    wins.append(env.game.iswin())
                    steps.append(env.steps)
                    i_word += 1
        
        self.agent.eps = eps
        return scores, wins, steps

    def log(self, train_stats, test_stats, game_output):
        # collected stats
        n_episodes_played, train_scores, train_wins, elapsed_time = train_stats
        test_scores, test_wins, test_steps = test_stats

        # simple aggregations
        train_win_rate = 100 * np.mean(train_wins)
        test_win_rate = 100 * np.mean(test_wins)
        test_mean_steps = np.mean(test_steps)
        test_mean_score = np.mean(test_scores)

        if self.detailed_logging:
            # top 3 hardest and easiest words
            hard_inds = np.argpartition(self.train_env_list[0].game.loses, kth=range(-3, 0))[range(-3, 0)]
            easy_inds = np.argpartition(self.train_env_list[0].game.wins, kth=range(-3, 0))[range(-3, 0)]
            hard_words = [self.test_word_list[i] for i in hard_inds]
            easy_words = [self.test_word_list[i] for i in easy_inds]

            # distribution of games by number of steps made
            n_letters = self.train_env_list[0].game.n_letters
            test_steps_distribution = np.bincount(test_steps, minlength=n_letters+1)[1:]
            tsd = ', '.join([f'({i}) {val}' for i, val in enumerate(test_steps_distribution, start=1)])
            
            # distribution of games by total reward gained
            test_reward_distribution = np.histogram(test_scores, bins=10)
            counts, bins = test_reward_distribution
            trd = ', '.join([f'[{bins[i]:.1f},{bins[i+1]:.1f}): {val}' for i, val in enumerate(counts)])

        # will be returned in the end of training
        def to_txt(txt, string):
            open(f'{self.nickname}/{txt}', 'a+').write(str(string)+',')
        to_txt('train_scores.txt', ','.join([str(a) for a in train_scores]))
        to_txt('train_win_rates.txt', train_win_rate)
        to_txt('test_mean_scores.txt', test_mean_score)
        to_txt('test_win_rates.txt', test_win_rate)
        
        # rewards distribution
        trtd = self.flush_reward_logs()

        self.n_episodes_played += n_episodes_played
        self.elapsed_time += elapsed_time

        wandb.log({
        "Train Win Rate": train_win_rate,
        "Train Mean Score": np.mean(train_scores),
        "Test Win Rate": test_win_rate,
        "Test Mean Steps": test_mean_steps,
        "Test Mean Score": test_mean_score,
        "Agent Epsilon": self.agent.eps,
        "Episodes Played": self.n_episodes_played,
        "Elapsed Time": self.elapsed_time
    })

        # to train logs
        message = '\t'.join([
            f'\nEpisodes: {self.n_episodes_played:4d}',
            f'Time: {self.elapsed_time:.0f} s',
            f'Agent Eps: {self.agent.eps:.2f}',
            f'Train Win Rate: {train_win_rate:.2f}%',
            f'Test Win Rate: {test_win_rate:.2f}%',
            f'Test Mean Steps: {test_mean_steps:.4f}',
           
        ])
        if self.detailed_logging:
            message += '\n\n' + '\t'.join([
                f'Hard Words: {", ".join(hard_words)}',
                f'Easy Words: {", ".join(easy_words)}',
            ]) + '\n' + '\n'.join([
                f'Test Games Distribution by Steps: {tsd}',
                f'Test Games Distribution by Reward: {trd}',
                f'Test Rewards Contributions: {trtd}'
            ])
        self.print(message, self.logs_output)

        # to game report
        message = '\n'.join([
                f'Test Win Rate: {sum(test_wins)} / {len(test_wins)} ({test_win_rate:.2f}%)',
                f'Test Mean Steps: {test_mean_steps:.4f}',
        ])
        if self.detailed_logging:
            message += '\n' + '\n'.join([
                f'Hard Words: {", ".join(hard_words)}',
                f'Easy Words: {", ".join(easy_words)}',
                f'Test Games Distribution by Steps: {tsd}',
                f'Test Games Distribution by Reward: {trd}',
                f'Test Rewards Contributions: {trtd}'
            ])
        open(game_output, 'a+').write(message + '\n')

    def flush_reward_logs(self):
        # collate dicts from all environments
        collated = defaultdict(int)
        total = 0
        for env in self.train_env_list:
            for key, val in env.reward_stats.items():
                collated[key] += val
                total += val
        
        # clear all reward stats
        for env in self.train_env_list:
            for key in env.reward_stats.keys():
                env.reward_stats[key] = 0

        # compute fraction of each reward type
        res = {}
        for key, val in collated.items():
            res[key] = val / total

        return ', '.join([f'{key}: {100*val:.2f}%' for key, val in res.items()])

    def print(self, message, output):
        print(message)
        open(output, 'a+').write(message + '\n')

    def test_initial(self):
        # collected stats
        test_scores, test_wins, test_steps = self.play_test(f'{self.nickname}/test-initial.txt')
        
        # simple aggregations
        test_win_rate = 100 * np.mean(test_wins)
        test_mean_steps = np.mean(test_steps)

        # distribution of games by number of steps made
        n_letters = self.train_env_list[0].game.n_letters
        test_steps_distribution = np.bincount(test_steps, minlength=n_letters+1)[1:]
        tsd = ', '.join([f'{i}: {val}' for i, val in enumerate(test_steps_distribution, start=1)])
        
        # distribution of games by total reward gained
        test_reward_distribution = np.histogram(test_scores, bins=10)
        counts, bins = test_reward_distribution
        trd = ', '.join([f'[{bins[i]:.1f},{bins[i+1]:.1f}): {val}' for i, val in enumerate(counts)])

        # rewards distribution
        trtd = self.flush_reward_logs()

        message = '\t'.join([
            f'Initial Stats. ',
            f'Test Win Rate: {test_win_rate:.2f}%',
            f'Test Mean Steps: {test_mean_steps:.4f}',
        ]) + '\n\n' + '\n'.join([
            f'Test Games Distribution by Steps: {tsd}',
            f'Test Games Distribution by Reward: {trd}',
            f'Test Rewards Distribution by Type: {trtd}'
        ])

        # print train stats
        self.print(message, self.logs_output)
    
def exp_with_action(
    action_type: Literal['vocabulary', 'letters', 'comb_letters', 'wagons'],
    rewards,
    eps_start, eps_end, eps_decay, rb_size=int(1e6), n_letters=5, n_steps=6,
    lr=5e-4, combine_method='concat', hidden_size=256, n_hidden_layers=1, alpha=0, negative_weights=False, positive_weights=False,
    logging_interval=None, fine_tune=False, backbone_path=None, *, method_name,
    data, n_envs, optimize_interval, agent_path, n_episodes, n_episodes_warm, **action_specs
):
    """(Non-generic) experiment configurations. Operates with RLFramework."""

    train_answers, test_answers, guesses_set = data
    guesses_list = list(guesses_set)
    
    # create train list of parallel games 
    train_env_list = []
    for _ in range(n_envs):
        env = Environment(
            rewards=rewards,
            wordle=Wordle(
                vocabulary=guesses_set, answers=train_answers,
                negative_weights=negative_weights,
                positive_weights=positive_weights
            ),
            state_instance=StateYesNo(n_letters=n_letters, n_steps=n_steps)
        )
        train_env_list.append(env)
    state_size = train_env_list[0].state.size

    # synchronize pointers of all env instances
    for env in train_env_list[1:]:
        env.game.wins = train_env_list[0].game.wins
        env.game.loses = train_env_list[0].game.loses

    if alpha == 0:
        replay_buffer = ReplayBuffer(state_size=state_size, buffer_size=rb_size)
    else:
        replay_buffer = PrioritizedReplayBuffer(state_size=state_size, alpha=alpha, buffer_size=rb_size)

    if action_type == 'vocabulary':
        action = ActionVocabulary(
            vocabulary=guesses_list
        )
    elif action_type == 'letters':
        action = ActionLetters(
            vocabulary=guesses_list,
            ohe_matrix=action_specs['ohe_matrix'],
            wordle_list=action_specs['wordle_list'],
        )
    elif action_type == 'comb_letters':
        action = ActionCombLetters(
            vocabulary=guesses_list,
            ohe_matrix=action_specs['ohe_matrix'],
            k=action_specs['k'],
            wordle_list=action_specs['wordle_list']
        )
    elif action_type == 'wagons':
        action = ActionWagons(
            vocabulary=guesses_list,
            ohe_matrix=action_specs['ohe_matrix'],
            k=action_specs['k'],
            wordle_list=action_specs['wordle_list']
        )

    # create agent with weights from `agent_path`
    agent = Agent(
        state_size=state_size,
        action_instance=action,
        replay_buffer=replay_buffer,
        optimize_interval=optimize_interval,
        agent_path=agent_path,
        lr=lr,
        model=OldQNetwork,
        combine_method=combine_method,
        hidden_size=hidden_size,
        n_hidden_layers=n_hidden_layers
    )

    if fine_tune:
        if agent_path is None:
            raise ValueError('Fine tune is possible only with provided weights')
        agent.fine_tune()
    
    if backbone_path is not None:
        agent.load_backbone(backbone_path)

    # to track experiments
    problem_name = f'{len(test_answers)}-{len(guesses_list)}'
    nickname = f'{method_name}-{problem_name}'

    # training and evaluating utilities
    exp = RLFramework(
        agent=agent,
        train_env=train_env_list,
        test_word_list=test_answers,
        play_batch_size=len(train_env_list),
        n_episodes=n_episodes,
        n_episodes_warm=n_episodes_warm,
        nickname=nickname,
        logging_interval=logging_interval,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay
    )

    exp.train()
    
    return exp.nickname

def train_test_split(n_guesses, overfit, guesses, indices, in_answers):
    guesses = np.array(guesses)
    guesses_cur = guesses[indices[:n_guesses]]
    
    train_indices = []
    test_indices = []
    for i_guess in indices[:n_guesses]:
        if i_guess in in_answers:
            test_indices.append(i_guess)
        else:
            train_indices.append(i_guess)

    if overfit:
        train_answers_cur = guesses[test_indices]
    else:
        train_answers_cur = guesses[train_indices]
    
    test_answers_cur = guesses[test_indices]

    print(
        f'guesses: {len(guesses_cur)}',
        f'train answers: {len(train_answers_cur)}',
        f'test answers: {len(test_answers_cur)}' + (' (overfit strategy)' if overfit else ''),
        sep='\n'
    )

    return list(train_answers_cur), list(test_answers_cur), set(guesses_cur)

np.random.seed(0)

wandb.init(
    project='world',
    config={
        "overfit": True,
        "n_episodes_warm": 100,
        "eps_start": 1,
        "eps_end": 0.05,
        "eps_decay": 0.9954,
        "alpha": 0.4,
        "rb_size": int(1e5),
        "n_envs": 8
    }
)

# load words lists and precompute some stuff
answers = Wordle._load_vocabulary('answers.txt', astype=list)
guesses = Wordle._load_vocabulary('guesses.txt', astype=list)
wordle_list = guesses

in_answers = []
for i, word in enumerate(guesses):
  if word in answers:
    in_answers.append(i)

indices = np.arange(len(guesses))
np.random.shuffle(indices)

# some configs
rewards = {'repeat':-0.1, 'B':0, 'Y':1, 'G':1, 'win':15, 'lose':-15, 'step':-5}
n_guesses = 4000    # solve subproblem with part of words
overfit = True      # train and test words are the same
ohe_matrix = ActionLetters._make_ohe(vocabulary=wordle_list)

data = train_test_split(n_guesses, overfit, guesses, indices, in_answers)

# start training
exp_with_action(
    'letters', rewards,
    ohe_matrix=ohe_matrix,
    wordle_list=wordle_list,
    data=data,
    
    n_episodes=int(9000000),
    n_episodes_warm=100,
    logging_interval=10000,
    
    eps_start=1,
    eps_end=0.05,
    eps_decay=0.9954,

    alpha=0.4,
    rb_size=int(1e5),
    method_name='deepa-run',
    
    n_envs=8,
    optimize_interval=8,

    agent_path=None
)

wandb.finish()

