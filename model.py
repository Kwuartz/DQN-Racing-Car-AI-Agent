from collections import namedtuple, deque
from itertools import count
import random
import pygame
import math

import torch
import torch.nn as nn
import torch.optim as optim

from config import BATCH_SIZE, DISCOUNT_FACTOR, TARGET_UPDATE_STRENGTH, LR, TRAINING_TIMESTEP, BACKGROUND_COLOUR, SCREEN_HEIGHT, SCREEN_WIDTH, TRACK_HEIGHT, EXPLORATION_DECAY, NETWORK_INPUT_SIZE, NETWORK_ACTION_SIZE, TRACK_WIDTH, FPS, MAX_TIMESTEPS, EXPLORATION_START, MODELS_PATH, EXPLORATION_END, RED_CAR_IMAGE, VISUALISATION_STEP, TRAINING_EPISODES, EXPERIENCE_CAPACITY
from cars import CarAgent

Transition = namedtuple("Transition", ("state", "action", "nextState", "reward"))

class ExperienceMemory:
    def __init__(self, maximumSize):
        # Deque for efficiency
        self.memory = deque([], maxlen=maximumSize)

    def getSize(self):
        # To check how many experiences are stored in the memory
        return len(self.memory)

    def addExperience(self, state, action, nextState, reward):
        # Add experience as a tuple
        self.memory.append((state, action, nextState, reward))

    def getBatch(self, batchSize):
        # Randomly sample experiences to create a batch
        return random.sample(self.memory, batchSize)

class NeuralNetwork(nn.Module):
    def __init__(self, inputs, outputs):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(inputs, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, outputs)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

class DQNTrainer:
    def __init__(self, game):
        self.game = game
        self.device = game.device

        # Initialising neural network
        self.policyNet = NeuralNetwork(NETWORK_INPUT_SIZE, NETWORK_ACTION_SIZE).to(self.device)
        self.targetNet = NeuralNetwork(NETWORK_INPUT_SIZE, NETWORK_ACTION_SIZE).to(self.device)
        self.targetNet.load_state_dict(self.policyNet.state_dict())

        # Initialisng optimiser and experience memory
        self.optimizer = optim.AdamW(self.policyNet.parameters(), lr=LR, amsgrad=True)
        self.memory = ExperienceMemory(EXPERIENCE_CAPACITY)
    
    def updateModel(self):
        if self.memory.getSize() >= BATCH_SIZE:
            # Sampling experiences
            experiences = self.memory.getBatch(BATCH_SIZE)

            # Split batch into states, actions, nextStates and rewards arrays
            states, actions, nextStates, rewards = zip(*experiences)

            # Convert each array into tensors so I can manipulate them using pytorch
            states = torch.cat(states)
            actions = torch.cat(actions)
            rewards = torch.cat(rewards)

            # Creating a mask of valid states so that invalid states can be filtered out
            nonFinalMask = torch.tensor([state is not None for state in nextStates], device=self.device, dtype=torch.bool)

            # Only including valid next states because some might be None because the agent might have crashed
            nextStates = torch.cat([state for state in nextStates if state is not None]) 

            # Calculate and gather Q-Values for each action
            stateActionQValues = self.policyNet(states)

            # Gather the Q-Values for the chosen actions
            selectedActionQValues = stateActionQValues.gather(1, actions.unsqueeze(1))

            # Calculate target Q-Values for the next state
            nextStateQValues = torch.zeros(BATCH_SIZE, NETWORK_ACTION_SIZE, device=self.device)
            with torch.no_grad():
                nextStateQValues[nonFinalMask] = self.targetNet(nextStates)
                
            # Separate the max Q-values for acceleration and turning
            nextStateQValues = nextStateQValues.max(1).values

            # Compute the expected Q-Values for both acceleration and turning
            expectedQValues = (nextStateQValues * DISCOUNT_FACTOR) + rewards

            # Compute Huber loss for both acceleration and turning
            huberLoss = nn.SmoothL1Loss()
            
            # Combine both losses
            loss = huberLoss(selectedActionQValues, expectedQValues.unsqueeze(1))

            # Backpropogation
            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_value_(self.policyNet.parameters(), 100)
            self.optimizer.step()

    def train(self):
        episode = 1

        spawnPoint, spawnAngle = self.game.track.getSpawnPosition()

        while episode <= TRAINING_EPISODES:
            agentCar = CarAgent(spawnPoint.x, spawnPoint.y, spawnAngle, RED_CAR_IMAGE, self.game.track, self.policyNet, self.device, True)
            
            state = agentCar.getState()

            episodeReward = 0
            for timeStep in range(MAX_TIMESTEPS):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.game.running = False

                if self.game.running == False:
                    return

                # Calculating exploration threshold using exponential decay
                explorationThreshold = EXPLORATION_END + (EXPLORATION_START - EXPLORATION_END) * math.exp(-EXPLORATION_DECAY * episode)

                # Retrieving new experience
                action, nextState, reward, episodeEnded = agentCar.update(TRAINING_TIMESTEP, explorationThreshold)
                episodeReward += reward

                # Converting to tensors
                action = torch.tensor([action], device=self.device, dtype=torch.long)
                reward = torch.tensor([reward], device=self.device)

                # Store experience in memory
                self.memory.addExperience(state, action, nextState, reward)

                # Move to the next state
                state = nextState

                # Update the policy network
                self.updateModel()

                # Soft update target network
                self.softUpdateTargetNetwork()

                if episodeEnded:
                    break

            self.game.trainingMenu(episode, timeStep, episodeReward, explorationThreshold)
            
            if (episode) % VISUALISATION_STEP == 0:
                exitTraining = self.game.visualizeEpisode()

                if exitTraining:
                    break

            episode += 1
                    
        modelFilePath = self.game.modelSaveMenu()
        if modelFilePath:
            torch.save(self.policyNet.state_dict(), MODELS_PATH + "/" + modelFilePath + ".model")

    def softUpdateTargetNetwork(self):
        targetNetStateDict = self.targetNet.state_dict()
        policyNetStateDict = self.policyNet.state_dict()

        for key in policyNetStateDict:
            targetNetStateDict[key] = policyNetStateDict[key] * TARGET_UPDATE_STRENGTH + targetNetStateDict[key] * (1.0 - TARGET_UPDATE_STRENGTH)

        self.targetNet.load_state_dict(targetNetStateDict)
