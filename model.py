import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

from config import BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, TRAINING_TIMESTEP, BACKGROUND_COLOUR, SCREEN_HEIGHT, SCREEN_WIDTH, TRACK_HEIGHT, TRACK_WIDTH

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
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(inputs, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, outputs)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

class DQNTrainer:
    def __init__(self, screen, track, carImage):
        self.screen = screen
        self.track = track
        self.carImage = image

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Network parameters
        inputSize = 6
        actionSize = 6

        self.policyNet = NeuralNetwork(inputSize, actionSize).to(self.device)
        self.targetNet = NeuralNetwork(inputSize, actionSize).to(self.device)
        self.targetNet.load_state_dict(self.policyNet.state_dict())

        self.optimizer = optim.AdamW(self.policyNet.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
    
    def optimizeModel(self):
        if len(self.memory) < self.BATCH_SIZE:
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
        actionBatch = torch.cat(batch.action)
        rewardBatch = torch.cat(batch.reward)

        # Calculate and gather Q-Values for each action
        stateActionQValues = policyNet(stateBatch)

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
            nextStateValues[nonFinalMask] = targetNet(nonFinalNextStates).max(1).values

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

    def train(self):
        steps = 0
        episodes = 50

        spawnPoint, spawnAngle = self.track.getSpawnPosition()

        for episodeIndex in range(episodes):
            agentCar = CarAgent(
                spawnPoint.x,
                spawnPoint.y,
                spawnAngle,
                self.carImage,
                self.policyNet,
                self.device,
                True
            )

            state = agentCar.getState(self.track)
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            for t in count():
                action, nextState, reward, terminated, truncated = agentCar.update(TRAINING_TIMESTEP, track, steps)
                steps += 1

                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

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

                if done:
                    print(f"Episode {episodeIndex + 1} finished after {t + 1} timesteps.")
                    break
            
            if episodeIndex % 10 == 0:
                self.visualizeEpisode()
    
    def softUpdateTargetNetwork(self):
        targetNetStateDict = self.targetNet.state_dict()
        policyNetStateDict = self.policyNet.state_dict()

        for key in policyNetStateDict:
            targetNetStateDict[key] = policyNetStateDict[key] * self.TAU + targetNetStateDict[key] * (1.0 - self.TAU)

        self.targetNet.load_state_dict(targetNetStateDict)

    def visualizeEpisode(self):
        spawnPoint, spawnAngle = self.track.getSpawnPosition()
        agentCar = CarAgent(spawnPoint.x, spawnPoint.y, spawnAngle, redCarImage, policyNet, device, training=False)

        clock = pygame.time.Clock()
        deltaTime = self.clock.tick(FPS) / 1000

        trackSurface = pygame.Surface((TRACK_WIDTH, TRACK_HEIGHT))
        self.track.draw(trackSurface)
        scaledTrackSurface = pygame.transform.scale(trackSurface, (SCREEN_WIDTH, SCREEN_HEIGHT))

        visualisationRunning = True
        while visualisationRunning:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Select best action
            agentCar.update(state)

            self.screen.fill(BACKGROUND_COLOUR)
            self.screen.blit(scaledTrackSurface, (0, 0))
            agentCar.draw(self.screen, pygame.Vector2(0, 0))
            
            pygame.display.flip()

            self.deltaTime = self.clock.tick(FPS) / 1000

    print("Visualization complete.")
