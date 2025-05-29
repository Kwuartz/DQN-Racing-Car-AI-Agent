from collections import deque
import random
import pygame
import numpy as np
import math

from config import BATCH_SIZE, DISCOUNT_FACTOR, TARGET_UPDATE_STRENGTH, LR, TRAINING_TIMESTEP, BACKGROUND_COLOUR, SCREEN_HEIGHT, SCREEN_WIDTH, TRACK_HEIGHT, TRACK_WIDTH, FPS, MAX_TIMESTEPS, RED_CAR_IMAGE, VISUALISATION_STEP, TRAINING_EPISODES, EXPERIENCE_CAPACITY
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
            np.random.randn(inputs, 128) * 0.1,
            np.random.randn(128, 128) * 0.1,
            np.random.randn(128, outputs) * 0.1
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
        self.layerInputs = []
        self.layerOutputs = []

        for layerIndex in range(numLayers):
            x = np.dot(x, self.weights[layerIndex]) + self.biases[layerIndex]
            self.layerInputs.append(x) # Used during backpropogation to calculate the derivatives of the activation functions from the previous layer

            if layerIndex < numLayers - 1: # No activation function on the output layer
                x = self.relu(x)

            self.layerOutputs.append(x) # Used to calculate derivatives wrt the weights of neurons

        return x

class DQNTrainer:
    def __init__(self, game):
        self.game = game

        # Input and output layer sizes
        self.inputSize = 6
        self.actionSize = 6
        
        # Initialising policy and target network
        self.policyNet = NeuralNetwork(self.inputSize, self.actionSize)
        self.targetNet = NeuralNetwork(self.inputSize, self.actionSize)

        # Copying the policy network to the target network
        self.targetNet.weights = [np.copy(weights) for weights in self.policyNet.weights]
        self.targetNet.biases = [np.copy(biases) for biases in self.policyNet.biases]

        self.memory = ExperienceMemory(EXPERIENCE_CAPACITY)

        self.episode = 0

    def huberLoss(self, actualY, targetY, threshold=1):
        difference = actualY - targetY
        absoluteDifference = np.abs(difference)

        # Squared loss below the threshold and absolute loss above the threshold
        loss = np.where(absoluteDifference <= threshold, 0.5 * absoluteDifference**2, threshold * (absoluteDifference - (threshold * 0.5)))

        return loss

    def huberLossDerivative(self, actualY, targetY, threshold=1):
        difference = actualY - targetY  
        absoluteDifference = np.abs(difference)

        # The derrivative is equal to the absolute difference below the threshold and -1 or 1 (assuming threshold is 1) above the threshold
        derrivative = np.where(absoluteDifference <= threshold, difference, threshold * np.sign(difference))

        return derrivative

    def reluDerivative(self, x):
        # Returns 0 if x is less than or equal to 0 and 1 if greater than 0
        return (x > 0).astype(float)

    def backpropagation(self, inputs, derivativeLoss):
        # To store gradients for gradient descent
        weightGradients = []
        biasGradients = []

        # Derrivatives of the loss wrt the outputs of the layer
        # These value start out being equal to the derrivatives of the loss wrt the outputs of the output layer 
        # This is because there is no activation function on the output layer
        dLdZ = derivativeLoss
        
        for layerIndex in reversed(range(len(self.policyNet.weights))): # Work through the layers backwards
            if layerIndex != 0: # If its not the input layer
                # Use the outputs of the previous layer to calculate the derrivatives wrt the weights
                # We use .T to transpose the inputs so that the shape of the input matrix lines up with the dLdZ matrix
                dLdW = np.dot(self.policyNet.layerOutputs[layerIndex - 1].T, dLdZ) / BATCH_SIZE
            else: # If its the input layer
                # Use the inputs to calculate the derrivatives wrt the weights
                dLdW = np.dot(inputs.T, dLdZ) / BATCH_SIZE

            # Derrivatives of biases
            dLdB = np.sum(dLdZ, axis=0) / BATCH_SIZE

            weightGradients.append(dLdW)
            biasGradients.append(dLdB)

            #  If its not the input layer
            if layerIndex != 0:  
                # Use the dZdA where A is the post-activation values of the previous layer (equal to the weights of the current layer) to find dLdA
                dLdA = np.dot(dLdZ, self.policyNet.weights[layerIndex].T)

                # Use the derivative of the activation function to find the dLdZ of the previous layer
                dLdZ = dLdA * self.reluDerivative(self.policyNet.layerInputs[layerIndex - 1])

        # Reverse the gradients because they were appended backwards
        weightGradients.reverse()
        biasGradients.reverse()

        return weightGradients, biasGradients

    def gradientDescent(self, weightGradients, biasGradients):
        for layerIndex in range(len(self.policyNet.weights)):
            # Retrieve gradients for current layer
            dLdW = weightGradients[layerIndex]
            dLdB = biasGradients[layerIndex]

            # Clip gradients to stabilise training
            dLdW = np.clip(dLdW, -1.0, 1.0)
            dLdB = np.clip(dLdB, -1.0, 1.0)

            # Update the weights and biases using our chosen learning rate value
            self.policyNet.weights[layerIndex] -= LR * dLdW
            self.policyNet.biases[layerIndex] -= LR * dLdB

    def adam(self):
        pass

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

            if np.any(validStateMask):
                nextStateQValues[validStateMask] = self.targetNet.forwardPass(nextStateBatch)
                
            # Select the max Q-values for acceleration and turning for the next state
            nextStateAccelerationQValues = np.max(nextStateQValues[:, :3], axis=1)
            nextStateTurningQValues = np.max(nextStateQValues[:, 3:], axis=1)

            # Calculate the expected Q-Values for both acceleration and turning
            expectedAccelerationQValues = (nextStateAccelerationQValues * DISCOUNT_FACTOR) + rewardBatch
            expectedTurningQValues = (nextStateTurningQValues * DISCOUNT_FACTOR) + rewardBatch

            # Calculate Huber loss derivative for both acceleration and turning
            accelerationLossDerivative = self.huberLossDerivative(selectedAccelerationQValues, expectedAccelerationQValues)
            turningLossDerivative = self.huberLossDerivative(selectedTurningQValues, expectedTurningQValues)
            
            # Insert both loss derivatives into a matrix of the output actions and set the derivatives of the actions that were not chosen to 0
            lossDerivative = np.zeros((BATCH_SIZE, self.actionSize))
            lossDerivative[np.arange(BATCH_SIZE), accelerationActions] = accelerationLossDerivative
            lossDerivative[np.arange(BATCH_SIZE), 3 + turningActions] = turningLossDerivative
            
            # Perform backpropogation to calculate gradients
            weightGradients, biasGradients = self.backpropagation(stateBatch, lossDerivative)

            # Perform gradient descent using calculated gradients
            self.gradientDescent(weightGradients, biasGradients)

    def train(self):
        steps = 0
        spawnPoint, spawnAngle = self.game.track.getSpawnPosition()

        self.episode_rewards = []
        self.episode_qvalues = []
        self.episode_epsilons = []

        # Stop the training after a specified number of episodes
        while self.episode <= TRAINING_EPISODES:
            # Update the training menu after each episode
            self.game.trainingMenu()

            self.episode += 1

            # Initialise the car agent
            agentCar = CarAgent(spawnPoint.x, spawnPoint.y, spawnAngle, RED_CAR_IMAGE, self.policyNet, True)
            
            state = agentCar.getState(self.game.track)

            # Stop the training episode after a certain amount of steps so the episode does not run indefinitely
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
                self.partialUpdateTargetNetwork()

                if terminated or truncated:
                    print(f"Episode {self.episode} finished after {timeStep} timesteps.")
                    break
            
            # Visualise the episode
            if (self.episode) % VISUALISATION_STEP == 0:
                self.game.visualizeEpisode()
    
    def partialUpdateTargetNetwork(self):
        for layerIndex in range(len(self.policyNet.weights)):
            self.targetNet.weights[layerIndex] = self.policyNet.weights[layerIndex] * TARGET_UPDATE_STRENGTH + self.targetNet.weights[layerIndex] * (1.0 - TARGET_UPDATE_STRENGTH)
            self.targetNet.biases[layerIndex] = self.policyNet.biases[layerIndex] * TARGET_UPDATE_STRENGTH + self.targetNet.biases[layerIndex] * (1.0 - TARGET_UPDATE_STRENGTH)