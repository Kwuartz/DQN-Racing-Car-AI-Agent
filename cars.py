from config import CAMERA_SCROLL_SPEED, SCREEN_WIDTH, SCREEN_HEIGHT, CHECKPOINT_REWARD, CRASH_REWARD, MAX_IDLE_TIMESTEPS, SPEED_REWARD, IDLE_REWARD, LAP_REWARD

import pygame
import torch
import random
import math

class Car:
    def __init__(self, x, y, direction, image):
        self.maxSpeed = 800
        self.acceleration = 300

        self.friction = 0.5
        self.brakeStrength = 3
        self.stoppingOffset = 50

        self.steerSpeed = 90
        self.steerCenterSpeed = 90

        self.x = x
        self.y = y

        self.lap = 0
        self.checkpointIndex = 0

        self.speed = 0
        self.direction = direction + 90
        self.wheelDirection = self.direction

        self.image = image
        self.rotatedImage = pygame.transform.rotate(self.image, -self.direction)
        self.mask = pygame.mask.from_surface(self.image)
        self.maskOffset = (0, 0)

        self.rect = image.get_rect()
        self.imageRect = self.rotatedImage.get_rect()
        self.rect.center = (x, y)
        self.imageRect.center = self.rect.center

    def handleInputs(self, deltaTime, acceleration, steerDirection):
        if acceleration == 1:
            if self.speed >= 0:
                self.speed = min(
                    self.speed + self.acceleration * deltaTime, self.maxSpeed)
            else:
                self.speed = min(self.speed + (-self.speed * self.brakeStrength *
                                 deltaTime + (deltaTime * self.stoppingOffset)), 0)
        elif acceleration == -1:
            if self.speed <= 0:
                self.speed = max(self.speed - (self.acceleration/2)
                                 * deltaTime, -self.maxSpeed/2)
            else:
                self.speed = max(self.speed - (self.speed * self.brakeStrength *
                                 deltaTime + (deltaTime * self.stoppingOffset)), 0)
        else:
            if self.speed > 0:
                self.speed = max(self.speed - (self.speed * self.friction *
                                 deltaTime + (deltaTime * self.stoppingOffset)), 0)
            else:
                self.speed = min(self.speed + (-self.speed * self.friction *
                                 deltaTime + (deltaTime * self.stoppingOffset)), 0)

        if steerDirection != 0:
            self.wheelDirection += steerDirection * self.steerSpeed * deltaTime
            self.wheelDirection = max(-45, min(self.wheelDirection, 45))
        else:
            if self.wheelDirection > 0:
                self.wheelDirection = max(
                    self.wheelDirection - self.steerCenterSpeed * deltaTime, 0)
            elif self.wheelDirection < 0:
                self.wheelDirection = min(
                    self.wheelDirection + self.steerCenterSpeed * deltaTime, 0)

    def moveCar(self, deltaTime, track):
        if self.speed != 0 and self.wheelDirection != 0:
            turningRadius = ((self.rect.height * 3.5) + (self.rect.height * 1.5) * (self.speed / self.maxSpeed)) / math.tan(math.radians(self.wheelDirection))
            angularVelocity = self.speed / turningRadius
            directionChange = math.degrees(angularVelocity * deltaTime)

            newImage = pygame.transform.rotate(self.image, -(self.direction + directionChange))
            newMask = pygame.mask.from_surface(newImage)

            newImageRect = newImage.get_rect()
            newImageRect.center = self.rect.center

            overlap = track.getOverlap(self.imageRect.x, self.imageRect.y, newMask)
            if not overlap:
                self.rotatedImage = newImage
                self.imageRect = newImageRect
                self.mask = newMask

                self.direction += directionChange

        xChange = self.speed * \
            math.sin(math.radians(self.direction)) * deltaTime
        yChange = self.speed * \
            math.cos(math.radians(self.direction)) * deltaTime

        overlapX = track.getOverlap(self.imageRect.x + xChange, self.imageRect.y, self.mask)
        if not overlapX:
            self.x += xChange
            self.rect.x = self.x
            self.imageRect.center = self.rect.center

        overlapY = track.getOverlap(self.imageRect.x, self.imageRect.y - yChange, self.mask)
        if not overlapY:
            self.y -= yChange
            self.rect.y = self.y
            self.imageRect.center = self.rect.center

        return (overlapX or overlapY)

    def collideCheckpoint(self, checkpoint):
        innerRect = pygame.Rect(0, 0, self.rect.width // 2, self.rect.height // 2)
        innerRect.center = self.rect.center

        if innerRect.clipline(checkpoint[0], checkpoint[1]):
            return True

        return False

    def update(self, deltaTime, acceleration, steerDirection, track):
        self.track = track
        self.handleInputs(deltaTime, acceleration, steerDirection)
        
        collision = self.moveCar(deltaTime, track)
        if collision:
            self.speed = 0

        nextCheckpoint = track.checkpoints[self.checkpointIndex]
        if self.collideCheckpoint(nextCheckpoint):
            self.checkpointIndex += 1

            if self.checkpointIndex > len(track.checkpoints) - 1:
                self.checkpointIndex = 0

            if self.checkpointIndex == 1:
                self.lap += 1

    def getCameraOffset(self, cameraOffset):
        xTrueOffset = self.x - SCREEN_WIDTH / 2
        yTrueOffset = self.y - SCREEN_HEIGHT / 2

        xOffset = (xTrueOffset - cameraOffset[0]) * CAMERA_SCROLL_SPEED
        yOffset = (yTrueOffset - cameraOffset[1]) * CAMERA_SCROLL_SPEED

        return cameraOffset + pygame.Vector2(xOffset, yOffset)

    def draw(self, screen, cameraOffset):
        screen.blit(self.rotatedImage, (self.imageRect.x - cameraOffset.x, self.imageRect.y - cameraOffset.y))

class CarAgent(Car):
    def __init__(self, x, y, direction, image, model, device, training):
        super().__init__(x, y, direction, image)
        
        # Distance sensors that will provide input to the neural network
        self.sensors= [
            -140,
            -105,
            -90,
            -75,
            -40,
        ]

        # Max distance of the sensors
        self.maxDistance = 500

        self.training = training
        if self.training:
            self.idleTimesteps = 0 

        self.model = model
        self.device = device

    def getDistances(self, track):
        distances = []

        for sensor in self.sensors:
            # Calculate angle
            sensorAngle = math.radians(self.direction + sensor)
            # Normalise angle
            directionVector = pygame.Vector2(math.cos(sensorAngle), math.sin(sensorAngle))

            # Default distance will be equal to the max distance checked by the sensor
            sensorDistance = self.maxDistance
            # Iterate until position of sensor is out of bounds
            # Iterating every 10 pixels to increase efficiency while sacrificing accuracy
            for distance in range(0, self.maxDistance, 10):
                position = self.imageRect.center + directionVector * distance

                if track.checkCollideAtPoint(position):
                    sensorDistance = distance
                    break
            
            # Normalise distance
            distances.append(sensorDistance / self.maxDistance)

        return distances

    def getState(self, track):
        # Normalise each input
        state = [
            self.speed / self.maxSpeed,
            *self.getDistances(track),
        ]

        return torch.tensor(state, dtype=torch.float32, device=self.device)

    def selectAction(self, state):
        # Calculate Q-Values
        QValues = self.model(state)
        
        # Get the best action
        selectedAction = QValues.argmax()

        # Convert index of best action to acceleration and turning value
        accelerationAction = (selectedAction // 3) - 1
        turningAction = (selectedAction % 3) - 1

        return accelerationAction, turningAction

    def update(self, deltaTime, track, explorationThreshold=0):
        # Get current state so it can be fed into the neural network
        state = self.getState(track)

        if self.training:
            reward = 0
            terminated = False
            truncated = False

            # Value 0 - 1 to be compared to exploration threshold
            sample = random.random()

            if sample > explorationThreshold:
                # Select best action
                accelerationAction, turningAction = self.selectAction(state)
            else:
                # Selecting random action
                accelerationAction = random.choice([-1, 0, 1])
                turningAction = random.choice([-1, 0, 1])
        else:
            # Select best action
            accelerationAction, turningAction = self.selectAction(state)

        # Update car position and direction according to inputs
        self.handleInputs(deltaTime, accelerationAction, turningAction)
        
        # Check if car has crashed
        collision = self.moveCar(deltaTime, track)
        if collision:
            self.speed = 0

            if self.training:
                reward += CRASH_REWARD
                terminated = True

        # Check whether car has crossed a checkpoint
        nextCheckpoint = track.checkpoints[self.checkpointIndex]
        if self.collideCheckpoint(nextCheckpoint):
            self.checkpointIndex += 1

            if self.training:
                reward += CHECKPOINT_REWARD
                self.idleTimesteps = 0

            # Check whether car has completed a lap
            if self.checkpointIndex > len(track.checkpoints) - 1:
                self.checkpointIndex = 0
            
            if self.checkpointIndex == 1:
                self.lap += 1
                
                # Truncate the episode if the agent finishes a lap
                if self.training:
                    reward += LAP_REWARD

                    if self.lap > 1:
                        truncated = True
        
        if self.training:
            # For use in bellman equation
            nextState = self.getState(track)

            # Convert back to action index
            action = (accelerationAction + 1) * 3 + (turningAction + 1)

            # Only add speed reward if speed is positive
            reward += max(0, self.speed) * SPEED_REWARD

            # If the agent does not get any reward for a long time then truncate the episode and punish it
            self.idleTimesteps += 1

            if self.idleTimesteps >= MAX_IDLE_TIMESTEPS:
                truncated = True
                reward += IDLE_REWARD

            return action, nextState, reward, terminated, truncated

        return collision
            
    def draw(self, screen, cameraOffset):
        screen.blit(self.rotatedImage, (self.imageRect.x - cameraOffset[0], self.imageRect.y - cameraOffset[1]))