import torch
import random
import numpy
from main import SnakeGame, Direction, Point
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 200_000
BATCH_SIZE = 5000
LR = 0.001


class Agent:

    def __init__(self):
        self.number_of_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(14, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_left = Point(head.x - 20, head.y)
        point_right = Point(head.x + 20, head.y)
        point_up = Point(head.x, head.y - 20)
        point_down = Point(head.x, head.y + 20)

        point_left2 = Point(head.x - 40, head.y)
        point_right2 = Point(head.x + 40, head.y)
        point_up2 = Point(head.x, head.y - 40)
        point_down2 = Point(head.x, head.y + 40)

        direction_left = game.direction == Direction.LEFT
        direction_right = game.direction == Direction.RIGHT
        direction_up = game.direction == Direction.UP
        direction_down = game.direction == Direction.DOWN

        state = [

            # Danger straight move
            direction_right and game.is_collision(point_right) or
            direction_left and game.is_collision(point_left) or
            direction_up and game.is_collision(point_up) or
            direction_down and game.is_collision(point_down),

            # Danger right move
            direction_right and game.is_collision(point_down) or
            direction_left and game.is_collision(point_up) or
            direction_up and game.is_collision(point_right) or
            direction_down and game.is_collision(point_left),

            # Danger left move
            direction_right and game.is_collision(point_up) or
            direction_left and game.is_collision(point_down) or
            direction_up and game.is_collision(point_left) or
            direction_down and game.is_collision(point_right),

            # Danger straight move
            direction_right and game.is_collision(point_right2) or
            direction_left and game.is_collision(point_left2) or
            direction_up and game.is_collision(point_up2) or
            direction_down and game.is_collision(point_down2),

            # Danger right move
            direction_right and game.is_collision(point_down2) or
            direction_left and game.is_collision(point_up2) or
            direction_up and game.is_collision(point_right2) or
            direction_down and game.is_collision(point_left2),

            # Danger left move
            direction_right and game.is_collision(point_up2) or
            direction_left and game.is_collision(point_down2) or
            direction_up and game.is_collision(point_left2) or
            direction_down and game.is_collision(point_right2),

            direction_right,
            direction_left,
            direction_up,
            direction_down,

            # Food location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return numpy.array(state, int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # List of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.number_of_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()

    while True:
        # get old State
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        new_state = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, new_state, done)

        # remember
        agent.remember(state_old, final_move, reward, new_state, done)

        # check if done
        if done:
            # train long memory
            game.reset()
            agent.number_of_games += 1
            agent.train_long_memory()

            if score > record:
                record = score

                agent.model.save()

            print('Game', agent.number_of_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.number_of_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
