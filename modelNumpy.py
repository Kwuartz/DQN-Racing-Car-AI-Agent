from collections import deque
from itertools import count
import random
import pygame
import numpy as np
import math

from config import BATCH_SIZE, GAMMA, TAU, LR, TRAINING_TIMESTEP, BACKGROUND_COLOUR, SCREEN_HEIGHT, SCREEN_WIDTH, TRACK_HEIGHT, TRACK_WIDTH, FPS, MAX_TIMESTEPS, RED_CAR_IMAGE, VISUALISATION_STEP, TRAINING_EPISODES
from cars import CarAgent


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

class NeuralNetwork:
    def __init__(self, inputs, outputs):
        # Initialising neural network with small random weights
        self.weights = [
            np.random.rand(inputs, 128) * 0.1,
            np.random.rand(128, 128) * 0.1,
            np.random.rand(128, outputs) * 0.1
        ]

        self.biases = [
            np.zeros(128),
            np.zeros(128),
            np.zeros(outputs)
        ]

    def relu(self, x): # Activation function
        return np.maximum(0, x)

    def forwardPass(self, x): # Forward propogation
        numLayers = len(self.weights)
        for layerIndex in range(numLayers):
            x = np.dot(x, self.weights[layerIndex]) + self.biases[layerIndex]

            if layerIndex < numLayers - 1: # No activation function on the output layer
                x = self.relu(x)

        return x

class DQNTrainer:
    def __init__(self, game):
        self.game = game

        # Input and output layer sizes
        self.inputSize = 7
        self.actionSize = 6
        
        # Initialising policy and target network
        self.policyNet = NeuralNetwork(self.inputSize, self.actionSize)
        self.targetNet = NeuralNetwork(self.inputSize, self.actionSize)

        # Copying the policy network to the target network
        self.targetNet.weights = [np.copy(weights) for weights in self.policyNet.weights]
        self.targetNet.biases = [np.copy(biases) for biases in self.policyNet.biases]

        self.memory = ExperienceMemory(10000)

        self.episode = 0

    def huberLoss(actualY, targetY, threshold=1):
        difference = targetY - actualY
        absoluteDifference = np.abs(difference)

        # Squared loss below the threshold and absolute loss above the threshold
        loss = np.where(absoluteDifference <= threshold, 0.5 * absoluteDifference^2, threshold * (absoluteDifference - (threshold * 0.5)))

        return loss

    def huberLossDerivative(self, actualY, targetY, threshold=1):
        difference = targetY - actualY
        absoluteDifference = np.abs(difference)

        # The derrivative is equal to the absolute difference below the threshold and -1 or 1 (assuming threshold is 1) above the threshold
        derrivative = np.where(absoluteDifference <= threshold, absoluteDifference, threshold * np.sign(difference))

        return derrivative

    def reluDerivative(self, x):
        # Returns 0 if x is less than or equal to 0 and 1 if greater than 0
        return (x > 0).astype(float)

    def backpropagation(self, inputs, derivativeLoss):
        # Derrivatives of the loss wrt the outputs of the layer
        # These value start out being equal to the derrivatives of the loss wrt the outputs of the output layer 
        # This is because there is no activation function on the output layer
        dLdZ = derivativeLoss

        for layerIndex in range(len(self.policyNet.weights) - 1, -1, -1): # Work through the layers backwards
            if layerIndex != 0: # If its not the input layer
                # Use the outputs of the previous layer to calculate the derrivatives wrt the weights
                # We use .T to transpose the inputs so that the shape of the input matrix lines up with the dLdZ matrix
                dLdW = np.dot(self.policyNet.layer_outputs[layerIndex - 1].T, dLdZ)
            else: # If its the input layer
                # Use the inputs to calculate the derrivatives wrt the weights
                dLdW = np.dot(inputs.T, dLdZ)

            # Derrivatives of biases
            dLdB = np.sum(dLdZ, axis=0)
            
            # Update the weights and biases using gradient descent and our chose learning rate value
            self.policyNet.weights[layerIndex] -= LR * dLdW
            self.policyNet.biases[layerIndex] -= LR * dLdB

            #  If its not the input layer
            if layerIndex != 0:  
                # Use the derrivative of the activation function to backpropogate dZ to the previous layer
                dLdZ = np.dot(dLdZ, self.policyNet.weights[layerIndex].T) * self.policyNet.relu_derivative(self.policyNet.layer_inputs[layerIndex - 1])

    def optimizeModel(self):
        if self.memory.getSize() >= BATCH_SIZE:
            experiences = self.memory.getBatch(BATCH_SIZE)
            
            # Split batch into states, actions, nextStates and rewards arrays
            states, actions, nextStates, rewards = zip(*experiences)

            # Convert each array into numpy arrays so I can manipulate them using numpy
            stateBatch = np.array(states)
            actionBatch = np.array(actions)
            rewardBatch = np.array(rewards)

            # Only including valid next states because some might be None because the agent might have crashed
            nextStateBatch = np.array([state for state in nextStates if state is not None]) 

            # Calculate and gather Q-Values for each action
            stateActionQValues = self.policyNet.forwardPass(stateBatch)

            # Separate Q-values for acceleration and turning
            accelerationQValues = stateActionQValues[:, :3]
            turningQValues = stateActionQValues[:, 3:]

            # Get the index of the action for each by shifting from -1, 0, 1 -> 0, 1, 2
            accelerationActions = actionBatch[:, 0] + 1
            turningActions = actionBatch[:, 1] + 1

            # Gather the Q-Values for the chosen actions
            selectedAccelerationQValues = accelerationQValues[np.arange(BATCH_SIZE), accelerationActions]
            selectedTurningQValues = turningQValues[np.arange(BATCH_SIZE), turningActions]

            # Creating a mask of valid states so that invalid states can be filtered out
            validStateMask = np.array([state is not None for state in nextStates])

            # Calculate Q-Values for the next state
            nextStateQValues = np.zeros((BATCH_SIZE, self.actionSize))
            nextStateQValues[validStateMask] = self.targetNet.forwardPass(nextStateBatch)
                
            # Select the max Q-values for acceleration and turning for the next state
            nextStateAccelerationQValues = np.max(nextStateQValues[:, :3], axis=1)
            nextStateTurningQValues = np.max(nextStateQValues[:, 3:], axis=1)

            # Calculate the expected Q-Values for both acceleration and turning
            expectedAccelerationQValues = (nextStateAccelerationQValues * GAMMA) + rewardBatch
            expectedTurningQValues = (nextStateTurningQValues * GAMMA) + rewardBatch

            # Calculate Huber loss derivative for both acceleration and turning
            accelerationLossDerivative = self.huberLossDerivative(selectedAccelerationQValues, expectedAccelerationQValues)
            turningLossDerivative = self.huberLossDerivative(selectedTurningQValues, expectedTurningQValues)
            
            # Insert both loss derivatives into a matrix of the output actions and set the derivatives of the actions that were not chosen to 0
            lossDerivative = np.zeros((BATCH_SIZE, self.actionSize))
            lossDerivative[np.arange(BATCH_SIZE), accelerationActions] = accelerationLossDerivative
            lossDerivative[np.arange(BATCH_SIZE), 3 + turningActions] = turningLossDerivative
            
            # Perform backpropogation
            self.backpropagation(states, lossDerivative)

    def train(self):
        steps = 0

        spawnPoint, spawnAngle = self.game.track.getSpawnPosition()

        while self.episode <= TRAINING_EPISODES:
            self.game.trainingMenu()

            self.episode += 1
            agentCar = CarAgent(spawnPoint.x, spawnPoint.y, spawnAngle, RED_CAR_IMAGE, self.policyNet, True)
            
            state = agentCar.getState(self.game.track)

            for timeStep in range(MAX_TIMESTEPS):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.game.running = False

                if self.game.running == False:
                    return

                steps += 1

                # Retrieving new transition
                action, nextState, reward, terminated, truncated = agentCar.update(TRAINING_TIMESTEP, self.game.track, steps)

                if terminated:
                    nextState = None

                # Store transition in memory
                self.memory.addExperience(state, action, nextState, reward)

                # Move to the next state
                state = nextState

                # Perform one step of optimisation on the policy network
                self.optimizeModel()

                # Soft update target network
                self.softUpdateTargetNetwork()

                if terminated or truncated:
                    print(f"Episode {self.episode} finished after {timeStep} timesteps.")
                    break
            
            if (self.episode) % VISUALISATION_STEP == 0:
                self.game.visualizeEpisode()
    
    def softUpdateTargetNetwork(self):
        def softUpdateTargetNetwork(self):
            for layerIndex in range(len(self.policyNet.weights)):
                self.targetNet.weights[layerIndex] = self.policyNet.weights[layerIndex] * TAU + self.targetNet.weights[layerIndex] * (1.0 - TAU)
                self.targetNet.biases[layerIndex] = self.policyNet.biases[layerIndex] * TAU + self.targetNet.biases[layerIndex] * (1.0 - TAU)