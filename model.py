from collections import namedtuple, deque
from itertools import count
import random
import pygame
import math

import torch
import torch.nn as nn
import torch.optim as optim

from config import BATCH_SIZE, GAMMA, TAU, LR, TRAINING_TIMESTEP, BACKGROUND_COLOUR, SCREEN_HEIGHT, SCREEN_WIDTH, TRACK_HEIGHT, TRACK_WIDTH, FPS, MAX_TIMESTEPS, RED_CAR_IMAGE, VISUALISATION_STEP
from cars import CarAgent

Transition = namedtuple("Transition", ("state", "action", "nextState", "reward"))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batchSize):
        return random.sample(self.memory, batchSize)

    def __len__(self):
        return len(self.memory)

class NeuralNetwork(nn.Module):
    def __init__(self, inputs, outputs):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(inputs, 128)
        self.norm = nn.LayerNorm(128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, outputs)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.norm(x)
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

class DQNTrainer:
    def __init__(self, game):
        self.game = game
        self.device = game.device

        # Network parameters
        inputSize = 5
        actionSize = 6

        self.policyNet = NeuralNetwork(inputSize, actionSize).to(self.device)
        self.targetNet = NeuralNetwork(inputSize, actionSize).to(self.device)
        self.targetNet.load_state_dict(self.policyNet.state_dict())

        self.optimizer = optim.AdamW(self.policyNet.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
    
    def optimizeModel(self):
        if len(self.memory) < BATCH_SIZE:
            return

        # Sampling transitions
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Removing final states
        nonFinalMask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.nextState)),
            device=self.device, 
            dtype=torch.bool
        )
        nonFinalNextStates = torch.cat([s for s in batch.nextState if s is not None])

        # Concatenating tensors
        stateBatch = torch.cat(batch.state)
        actionBatch = torch.stack(batch.action)
        rewardBatch = torch.cat(batch.reward)

        # Calculate and gather Q-Values for each action
        stateActionQValues = self.policyNet(stateBatch)

        # Separate Q-values for acceleration and turning
        accelerationQValues = stateActionQValues[:, :3]
        turningQValues = stateActionQValues[:, 3:]

        # Get the action for each and shift from -1, 0, 1 -> 0, 1, 2
        accelerationActions = actionBatch[:, 0] + 1
        turningActions = actionBatch[:, 1] + 1

        # Gather the Q-Values for the chosen actions
        selectedAccelerationQValues = accelerationQValues.gather(1, accelerationActions.unsqueeze(1))
        selectedTurningQValues = turningQValues.gather(1, turningActions.unsqueeze(1))

        # Combine the Q-values for both actions (acceleration and turning)
        stateActionQValues = selectedAccelerationQValues + selectedTurningQValues

        # Calculate target Q-Values for the next state
        nextStateValues = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            nextStateValues[nonFinalMask] = self.targetNet(nonFinalNextStates).max(1).values

        # Compute the expected Q-Values
        expectedStateActionValues = (nextStateValues * GAMMA) + rewardBatch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(stateActionQValues, expectedStateActionValues.unsqueeze(1))

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policyNet.parameters(), 100)
        self.optimizer.step()

    def train(self, episodes):
        steps = 0

        spawnPoint, spawnAngle = self.game.track.getSpawnPosition()

        for episodeIndex in range(episodes):
            agentCar = CarAgent(spawnPoint.x, spawnPoint.y, spawnAngle, RED_CAR_IMAGE, self.policyNet, self.device, True)
            
            state = agentCar.getState(self.game.track)
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            for timeStep in range(MAX_TIMESTEPS):
                steps += 1
                action, nextState, reward, terminated, truncated = agentCar.update(TRAINING_TIMESTEP, self.game.track, steps)
        
                action = torch.tensor(action, device=self.device, dtype=torch.long)
                reward = torch.tensor([reward], device=self.device)
                if terminated:
                    nextState = None
                else:
                    nextState = torch.tensor(nextState, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Store transition in memory
                self.memory.push(state, action, nextState, reward)

                # Move to the next state
                state = nextState

                # Perform one step of optimisation on the policy network
                self.optimizeModel()

                # Soft update target network
                self.softUpdateTargetNetwork()

                if terminated or truncated:
                    print(f"Episode {episodeIndex + 1} finished after {timeStep} timesteps.")
                    break
            
            if (episodeIndex + 1) % VISUALISATION_STEP == 0:
                self.game.visualizeEpisode()
            
            

    
    def softUpdateTargetNetwork(self):
        targetNetStateDict = self.targetNet.state_dict()
        policyNetStateDict = self.policyNet.state_dict()

        for key in policyNetStateDict:
            targetNetStateDict[key] = policyNetStateDict[key] * TAU + targetNetStateDict[key] * (1.0 - TAU)

        self.targetNet.load_state_dict(targetNetStateDict)
