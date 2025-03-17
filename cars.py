from config import CAMERA_SCROLL_SPEED, ASSETS_PATH, SCREEN_WIDTH, SCREEN_HEIGHT, CHECKPOINT_REWARD, LAP_REWARD, CRASH_REWARD, EPS_START, EPS_END, EPS_DECAY

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

        self.lap = 1
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
                self.lap += 1

    def getCameraOffset(self, cameraOffset):
        xTrueOffset = self.x - SCREEN_WIDTH / 2
        yTrueOffset = self.y - SCREEN_HEIGHT / 2

        xOffset = (xTrueOffset - cameraOffset[0]) * CAMERA_SCROLL_SPEED
        yOffset = (yTrueOffset - cameraOffset[1]) * CAMERA_SCROLL_SPEED

        return cameraOffset + pygame.Vector2(xOffset, yOffset)

    def getDistances(self, screen, cameraOffset, maxDistance=200):
        distances = []

        self.sensors= [
            -180,
            -135,
            -45,
            0,
        ]

        for sensor in self.sensors:
            sensorAngle = math.radians(self.direction + sensor)
            directionVector = pygame.Vector2(math.cos(sensorAngle), math.sin(sensorAngle))

            sensorDistance = maxDistance
            for distance in range(maxDistance):
                position = self.imageRect.center + directionVector * distance
                screen.set_at((round(position.x - cameraOffset.x), round(position.y - cameraOffset.y)), (255, 0, 0))

                if self.track.checkCollideAtPoint(position):
                    sensorDistance = distance
                    break
            distances.append(sensorDistance)
        
        return distances

    def draw(self, screen, cameraOffset):
        screen.blit(self.rotatedImage, (self.imageRect.x - cameraOffset.x, self.imageRect.y - cameraOffset.y))
        self.getDistances(screen, cameraOffset)

class CarAgent(Car):
    def __init__(self, x, y, direction, image, model, device):
        super().__init__(x, y, direction, image)
        
        self.sensors= [
            -180,
            -135,
            -45,
            0,
        ]

        self.model = model
        self.device = device

    def getDistances(self, track, maxDistance=600):
        distances = []

        for sensor in self.sensors:
            sensorAngle = math.radians(self.direction + sensor)
            directionVector = pygame.Vector2(math.cos(sensorAngle), math.sin(sensorAngle))

            sensorDistance = maxDistance
            for distance in range(maxDistance):
                position = self.imageRect.center + directionVector * distance

                if track.checkCollideAtPoint(position):
                    sensorDistance = distance
                    break
                
            distances.append(sensorDistance)
        
        return distances

    def getState(self, track):
        return [
            self.speed,
            self.direction,
            *self.getDistances(track)
        ]

    def selectAction(self, state):
        with torch.no_grad():
            # Calculate Q-Values
            QValues = self.model(state)
            
            # Seperate Q-Values for acceleration and turning
            accelerationQValues = QValues[:, :3]
            turningQValues = QValues[:, 3:]

            # Get the best action for each and shift from 0, 1, 2 -> -1, 0, 1
            accelerationAction = accelerationQValues.max(1).indices.item() - 1
            turningAction = turningQValues.max(1).indices.item() - 1

            return accelerationAction, turningAction

    def update(self, deltaTime, track, steps=None):
        state = self.getState(track)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        if steps:
            reward = 0
            terminated = False

            # Value 0 - 1 to be compared to epsilon threshold
            sample = random.random()

            # Calculating epsilon threshold
            epsThreshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps / EPS_DECAY)

            steps += 1
            if sample > epsThreshold:
                accelerationAction, turningAction = self.selectAction(state)
            else:
                accelerationAction = random.choice([-1, 0, 1])
                turningAction = random.choice([-1, 0, 1])
        else:
            accelerationAction, turningAction = self.selectAction(state)

        self.handleInputs(deltaTime, accelerationAction, turningAction)
        
        collision = self.moveCar(deltaTime, track)
        if collision:
            self.speed = 0

            if steps:
                reward += CRASH_REWARD
                terminated = True

        nextCheckpoint = track.checkpoints[self.checkpointIndex]
        if self.collideCheckpoint(nextCheckpoint):
            self.checkpointIndex += 1
            
            if steps:
                reward += CHECKPOINT_REWARD

            if self.checkpointIndex > len(track.checkpoints) - 1:
                self.checkpointIndex = 0
                self.lap += 1

                if steps:
                    reward += LAP_REWARD
                    terminated = True

        if steps:
            nextState = self.getState(track)
            action = accelerationAction, turningAction
            return action, nextState, reward, terminated
            
    def draw(self, screen, cameraOffset):
        screen.blit(self.rotatedImage, (self.imageRect.x - cameraOffset[0], self.imageRect.y - cameraOffset[1]))