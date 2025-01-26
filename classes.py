import pygame
import math

class Car:
  maxSpeed = 800
  acceleration = 400
  friction = 0.25
  brakeStrength = 2
  stoppingOffset = 100
  steerSpeed = 90
  steerCenterSpeed = 90

  width = 60
  height = 80

  def __init__(self, x, y, direction, image):
    self.x = x
    self.y = y
    self.speed = 0
    self.wheelDirection = direction
    self.direction = direction
    self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
    self.image = pygame.transform.scale(image, (self.width, self.height))

  def update(self, deltaTime, acceleration, steerDirection):
    if acceleration == 1:
      if self.speed >= 0:
        self.speed = min(self.speed + self.acceleration * deltaTime, self.maxSpeed)
      else:
        self.speed = min(self.speed + (-self.speed * self.brakeStrength * deltaTime + (deltaTime * self.stoppingOffset)), 0)     
    elif acceleration == -1:
      if self.speed <= 0:
        self.speed = max(self.speed - (self.acceleration/2) * deltaTime, -self.maxSpeed/2)
      else:
        self.speed = max(self.speed - (self.speed * self.brakeStrength * deltaTime + (deltaTime * self.stoppingOffset)), 0)
    else:
      if self.speed > 0:
        self.speed = max(self.speed - (self.speed * self.friction * deltaTime + (deltaTime * self.stoppingOffset)), 0)
      else:
        self.speed = min(self.speed + (-self.speed * self.friction * deltaTime + (deltaTime * self.stoppingOffset)), 0)
    
    if steerDirection != 0:
      self.wheelDirection += steerDirection * self.steerSpeed * deltaTime
      self.wheelDirection = max(-45, min(self.wheelDirection, 45))
    else:
      if self.wheelDirection > 0:
        self.wheelDirection = max(self.wheelDirection - self.steerCenterSpeed * deltaTime, 0)
      elif self.wheelDirection < 0:
        self.wheelDirection = min(self.wheelDirection + self.steerCenterSpeed * deltaTime, 0)
    
    if self.speed != 0 and self.wheelDirection != 0:
      turningRadius = (self.height * 2) / math.tan(math.radians(self.wheelDirection))
      angularVelocity = self.speed / turningRadius
      self.direction += math.degrees(angularVelocity * deltaTime)


    self.x += self.speed * math.sin(math.radians(self.direction)) * deltaTime
    self.y -= self.speed * math.cos(math.radians(self.direction)) * deltaTime

    self.rect.x = self.x
    self.rect.y = self.y

  def draw(self, screen):
    rotatedImage = pygame.transform.rotate(self.image, -self.direction)
    imageRect = rotatedImage.get_rect()
    imageRect.center = self.rect.center
    screen.blit(rotatedImage, imageRect)