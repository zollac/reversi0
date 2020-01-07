import time
import math
import numpy as np
import tensorflow as tf
from itertools import count
from tensorflow.keras.models import model_from_json
from typing import List
from scipy.special import softmax


class Config(object):
    """class containing hyperparameters"""
    def __init__(self):
        # the following hyperparameters are rather arbitrary and
        # haven't been tuned to result in efficient training

        ### Self-Play

        self.num_games = 1000 # number of self play games generated
        self.num_sampling_moves = 30 # when to stop softmax sampling and choose moves greedily
        self.num_simulations = 100 # number of simulations in MCTS, originall 800

        # Root prior exploration noise.
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Training
        self.num_training_cycles = 100 # number of game generation + training cycles

        self.training_steps = 100 # int(700e3)
        self.window_size = self.num_games # int(1e6)
        self.batch_size = 1024 # 4096

        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.learning_rate = 2e-2


class Node(object):
    """search node"""

    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0: return 0

        return self.value_sum / self.visit_count

# internal representation of the initial board position as a string

initial = (
    '         \n' #  0 -  9
    ' ........\n' # 10 - 19
    ' ........\n' # 20 - 29
    ' ........\n' # 30 - 39
    ' ...10...\n' # 40 - 49
    ' ...01...\n' # 50 - 59
    ' ........\n' # 60 - 69
    ' ........\n' # 70 - 79
    ' ........\n' # 80 - 89
    '         \n' # 90 - 99
)

# list of indices of squares within the board in the internal board representation
possible_actions = (11, 12, 13, 14, 15, 16, 17, 18,
                    21, 22, 23, 24, 25, 26, 27, 28,
                    31, 32, 33, 34, 35, 36, 37, 38,
                    41, 42, 43, 44, 45, 46, 47, 48,
                    51, 52, 53, 54, 55, 56, 57, 58,
                    61, 62, 63, 64, 65, 66, 67, 68,
                    71, 72, 73, 74, 75, 76, 77, 78,
                    81, 82, 83, 84, 85, 86, 87, 88)

# internal representation of the 8 directions N, NE, E, SE, S, SW, W, NW
directions = (-10, -9, 1, 11, 10, 9, -1, -11)

class Game(object):
    """Class representing a single game. history attribute contains a list of tuples of
        a board position, and the move that led to it"""

    def __init__(self, history=None):
        self.history = history or [(initial, None)]
        self.child_visits = []
        self.num_actions = 64  # action space size for reversi

    def legal_actions(self):
        # returns a generator of legal actions in the current position
        board = self.history[-1][0]
        for i, p in enumerate(board):
            if p != '.': continue
            for d in directions:
                done = False

                for j in count(i+d, d):
                    q = board[j]

                    # if we got off the board or encountered an empty square, break out of the ray
                    if q in ' \n.': break

                    # if we encounter the opponent's stone, continue with the next element of ray
                    elif q != str(self.to_play()): continue

                    # q now has to be our stone, if we are not right next to i, yield i
                    elif j != i+d:
                        yield(i)
                        done = True

                    break

                if done: break

    def apply(self, action):
        # applies action to the current position and updates history
        # passing is signalled by None action

        board = self.history[-1][0]

        # copy the board
        new_board = board

        # if not passing
        if action != None:
            player = self.to_play()
            put = lambda board, i: board[:i] + str(player) + board[i+1:]

            # put a stone to action
            new_board = put(new_board, action)
            for d in directions:
                for j in count(action+d, d):
                    q = board[j]
                    if q in ' \n.': break
                    elif q != str(player): continue
                    elif j != action+d:
                        # have to go back until square and recolor each square
                        for i in range(j-d, action, -d):
                            new_board = put(new_board, i)
                    break

        self.history.append((new_board, action))


    def terminal(self):
        # terminal test: no legal moves for either side
        try:
            next(self.legal_actions())
        except StopIteration:
            # no legal moves so pass
            self.apply(None)

            try:
                next(self.legal_actions())
            except StopIteration:

                # no legal move so game ended, need to delete pass move from history
                del self.history[-1]
                return True

            # there is a legal move, need to delete pass move from history
            del self.history[-1]

        return False

    def terminal_value(self, to_play):
        # returns the terminal value of the game
        # 1 if white won, 0 if black won and 0.5 for a draw
        blacks, whites = 0, 0
        for p in self.history[-1][0]:
            if p == '0': blacks += 1
            elif p == '1': whites += 1

        if whites > blacks: return 1
        elif whites < blacks: return 0
        else: return 0.5

    def clone(self):
        return Game(list(self.history))

    def store_search_statistics(self, root):
        sum_visits = sum(child.visit_count for child in root.children.values())
        self.child_visits.append([
            root.children[possible_actions[a]].visit_count / sum_visits if possible_actions[a] in root.children else 0
            for a in range(self.num_actions)
        ])

    def make_image(self, state_index: int):
        # Game specific feature planes.
        # since the neural net won't encounter terminal states
        # history is unimportant in reversi, similarly,
        # additional information (who's next) is not needed
        # with this simplification, we don't need to worry about zero planes for t<1
        # returns two binary 8x8 feature planes containing the two player's stones
        # of shape (1, 8, 8, 2)
        board = self.history[state_index][0].split()
        player1 = str(self.to_play())
        player2 = str((self.to_play() + 1) % 2)
        board1, board2 =  [[] for i in range(8)], [[] for i in range(8)]
        for i, row in enumerate(board):
            for q in row:
                if q == '.':
                    board1[i].append(0.0)
                    board2[i].append(0.0)
                elif q == player1:
                    board1[i].append(1.0)
                    board2[i].append(0.0)
                elif q == player2:
                    board2[i].append(1.0)
                    board1[i].append(0.0)

        return np.expand_dims(np.array([board1, board2], dtype='float32').reshape((8,8,2)), axis=0)


    def make_target(self, state_index: int):
        # returns target value, policy for supervisied learning
         return (self.terminal_value((state_index + 1) % 2),
                self.child_visits[state_index])

    def to_play(self):
        # current player, black is starting
        return (len(self.history) + 1) % 2


class ReplayBuffer(object):
    # stores self-play games

    def __init__(self, config):
        # window_size is set to num_games, in config so
        # replay buffer is generated by the latest network
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self):
        # Sample uniformly across positions.
        move_sum = float(sum(len(g.history) for g in self.buffer))
        games = np.random.choice(
            self.buffer,
            size=self.batch_size,
            p=[len(g.history) / move_sum for g in self.buffer])
        # the -1 in (len(g.history)-1) is needed, because g.history is one item longer
        # than Node(0).child_visits (called by make_target), as store_search_statistics
        # is not run on the starting position (which is the first element of game.history)
        game_pos = [(g, np.random.randint(len(g.history)-1)) for g in games]
        return [(g.make_image(i), g.make_target(i)) for (g, i) in game_pos]


class Network(object):
    def __init__(self):
        # body is made up of 3 conv2D blocks of 16 filters of size 3x3, and relu activation
        # network then splits into a value and policy head, the value head contains an additional
        # conv2D block of 1 filter and kernel size 1x1, a dense layer of size 64 and relu activation
        # and a tanh layer to a scalar ouputting the value prediction
        # the policy head contains conv2D layer of two filters of kernel size 1x1 and a single
        # linear layer outputting a 64 dimensional policy prediction
        # all conv2D layers have stride 1 and "same" padding
        # all layers are L_2 regularized
        # no batch norm, no skip connections since this is a shallow network

        filters_num = 16

        regularizer = tf.keras.regularizers.l2(0.01)
        body_input = tf.keras.Input(shape=(8,8,2))
        body1 = tf.keras.layers.Conv2D(filters_num, (3, 3), strides=1, padding='same', activation='relu', \
                                       input_shape=(8, 8, 2), kernel_regularizer=regularizer, name="body1")(body_input)

        body2 = tf.keras.layers.Conv2D(filters_num, (3, 3), strides=1, padding='same', activation='relu', \
                                       kernel_regularizer=regularizer, name="body2")(body1)

        body_out = tf.keras.layers.Conv2D(filters_num, (3, 3), strides=1, padding='same', activation='relu', \
                                          kernel_regularizer=regularizer, name="body_out")(body2)


        value1 = tf.keras.layers.Conv2D(1, (1, 1), strides=1, padding='same', activation='relu', \
                                        kernel_regularizer=regularizer, name="value1")(body_out)

        value2 = tf.keras.layers.Flatten()(value1)

        value3 = tf.keras.layers.Dense(64, activation='relu', \
                                       kernel_regularizer=regularizer, name="value3")(value2)

        value_out = tf.keras.layers.Dense(1, activation='tanh', \
                                          kernel_regularizer=regularizer, name="value_out")(value3)



        policy1 = tf.keras.layers.Conv2D(2, (1, 1), strides=1, padding='same', activation='relu', \
                                         kernel_regularizer=regularizer, name="policy1")(body_out)

        policy2 = tf.keras.layers.Flatten()(policy1)

        policy_out = tf.keras.layers.Dense(8*8, activation='linear', name="policy_out")(policy2)


        self.model = tf.keras.Model(inputs=[body_input], outputs=[value_out, policy_out])


    def inference(self, image):
        # returns the value and policy predictions of the network
        prediction = self.model.predict(image)
        value = (float(prediction[0]) + 1) / 2 # shifting the predicted value into the [0, 1] range
        policy = {possible_actions[i] : float(pol) for i, pol in enumerate(prediction[1][0])}
        return (value, policy)


class UniformNetwork(object):
    # default random prediction for first set of self-play games

    def inference(self, image):
        value = 0.5
        policy = {possible_actions[i] : 1.0/64 for i in range(64)}
        return (value, policy)


class SharedStorage(object):
    # stores the trained networks

    def __init__(self):
        self._networks = {}

    def latest_network(self) -> Network:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            return UniformNetwork()  # policy -> uniform, value -> 0.5

    def save_network(self, step: int, network: Network):
        self._networks[step] = network



# a training cycle consist of two parts, the first generates config.num_games games
# using the latest network in the shared storage and the second traines the network
# using supervised learning on the positions generated in the first part.
# value and policy are adjusted to reflect the terminal value of the game
# and the move chosen using Monte Carlo tree search
def reversi0(config: Config):
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)

    for i in range(config.num_training_cycles):
        print("launching cycle " + str(i+1) + "/" + str(config.num_training_cycles))
        run_selfplay(config, storage, replay_buffer)
        train_network(config, storage, replay_buffer)

    return storage.latest_network()


##################################
####### Part 1: Self-Play ########



# generates games using self-play with the latest network and writes them to the shared replay buffer
def run_selfplay(config: Config, storage: SharedStorage, replay_buffer: ReplayBuffer):
    network = storage.latest_network()
    print("generating games")
    for i in range(config.num_games):
        game = play_game(config, network)
        replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: Config, network: Network):
    game = Game()
    while not game.terminal():
        action, root = run_mcts(config, game, network)
        game.apply(action)
        game.store_search_statistics(root)

    return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run config.num_simulations simulations, always starting
# at the root of the search tree and traversing the tree according to the UCB formula
# until we reach a leaf node.
def run_mcts(config: Config, game: Game, network: Network):
    root = Node(0)
    # expands root by adding its children with priors calculated for Network
    evaluate(root, game, network)
    add_exploration_noise(config, root)

    for _ in range(config.num_simulations):
        node = root
        scratch_game = game.clone()
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node)
            scratch_game.apply(action)
            search_path.append(node)

        value = evaluate(node, scratch_game, network)
        backpropagate(search_path, value, scratch_game.to_play())
    return select_action(config, game, root), root


# select action according to visit_counts using softmax sampling for the
# first config.num_sampling_moves and greedily for the rest
def select_action(config: Config, game: Game, root: Node):
    visit_counts = [(child.visit_count, action)
                    for action, child in root.children.items()]
    if visit_counts == []:
        return None
    else:
        if len(game.history) < config.num_sampling_moves:
            _, action = softmax_sample(visit_counts)
        else:
            _, action = max(visit_counts)
    return action


# softmax sampling
def softmax_sample(visit_counts):
    visits = [element[0] for element in visit_counts]
    dist = softmax(visits)
    sampled_index = np.random.choice(len(visits), p=dist)
    return visit_counts[sampled_index]


# Select the child with the highest UCB score.
def select_child(config: Config, node: Node):
    _, action, child = max((ucb_score(config, node, child), action, child)
                            for action, child in node.children.items())
    return action, child



# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: Config, parent: Node, child: Node):
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                    config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = child.value()
    return prior_score + value_score


# We use the neural network to obtain a value and policy prediction.
def evaluate(node: Node, game: Game, network: Network):
    value, policy_logits = network.inference(game.make_image(-1))

    # Expand the node.
    node.to_play = game.to_play()
    policy = {a: math.exp(policy_logits[a]) for a in game.legal_actions()}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)
    return value


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play):
    for node in search_path:
        node.value_sum += value if node.to_play == to_play else (1 - value)
        node.visit_count += 1



# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: Config, node: Node):
    actions = node.children.keys()
    noise = np.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


######### End Self-Play ##########
##################################

##################################
####### Part 2: Training #########

# saves network to file
def save_network(network, filename=None):

    if not filename:
        filename = time.strftime("%Y%m%d-%H%M%S")

    model_json = network.model.to_json()
    with open(filename, "w") as json_file:
        json_file.write(model_json)

    network.model.save_weights("model.h5")

# trains the network on batches of positions from self-play
def train_network(config: Config, storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
    network = Network()
    optimizer = tf.train.MomentumOptimizer(config.learning_rate, config.momentum)

    for i in range(config.training_steps):
        print("in " + str(i+1) + "/" + str(config.training_steps) + " training")
        batch = replay_buffer.sample_batch()
        update_weights(optimizer, network, batch, config.weight_decay)

    save_network(network)
    storage.save_network(config.training_steps, network)


# calculates value and policy losses and fits the network model
def update_weights(optimizer: tf.train.Optimizer, network: Network, batch,
                   weight_decay: float):
    value_loss = tf.keras.losses.MeanSquaredError()
    policy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    network.model.compile(optimizer, loss=[value_loss, policy_loss])
    batch_size = len(batch)
    image_batch = np.array([image for (image, (_, _)) in batch]).reshape(batch_size,8,8,2)
    # shifting back the target values to the neural net range of [-1, 1]
    target_value_batch = np.array([2*target_value-1 for (_, (target_value, target_policy)) in batch])
    target_policy_batch = np.array([target_policy for (_, (target_value, target_policy)) in batch])
    network.model.fit(x=image_batch, y=[target_value_batch, target_policy_batch], batch_size=batch_size)



######### End Training ###########
##################################
