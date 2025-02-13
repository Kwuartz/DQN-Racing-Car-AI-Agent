from config import FPS, SCREEN_WIDTH, SCREEN_HEIGHT
import pygame
import math

class Game:
    def __init__(self, training = False):
        self.training = training

        pygame.init()
        
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Racing Game")

        self.clock = pygame.time.Clock()
        self.running = True
        self.deltaTime = 1 / self.FPS

        self.blueCarImage = pygame.image.load("Assets/BlueCar.png")
        self.redCarImage = pygame.image.load("Assets/RedCar.png")

        if not training:
            self.playerCar = Car(300, 300, 0, self.blueCarImage)

        self.agentCar= CarAgent(300, 300, 0, self.blueCarImage)

    def handleEvents(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self):
        if not self.training:
            acceleration = 0
            turnDirection = 0

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                acceleration += 1
            if keys[pygame.K_s]:
                acceleration -= 1
            if keys[pygame.K_d]:
                turnDirection += 1
            if keys[pygame.K_a]:
                turnDirection -= 1

            self.playerCar.update(self.deltaTime, acceleration, turnDirection)

        self.agentCar.update()

    def render(self):
        self.screen.fill((0, 0, 0))

        if not self.training:
            self.playerCar.draw(self.screen)
            self.agentCar.draw(self.screen)

        pygame.display.flip()

    def run(self):
        while self.running:
            self.handleEvents()
            self.update()

            if not self.training:
                self.render()
            
            self.deltaTime = self.clock.tick(self.FPS) / 1000

        pygame.quit()

class Car:
    def __init__(self, x, y, direction, image):
        self.maxSpeed = 800
        self.acceleration = 400

        self.friction = 0.25
        self.brakeStrength = 2
        self.stoppingOffset = 100

        self.steerSpeed = 90
        self.steerCenterSpeed = 90

        self.width = 60
        self.height = 80

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

        if self.speed != 0 and self.wheelDirection != 0:
            turningRadius = (self.height * 2) / \
                math.tan(math.radians(self.wheelDirection))
            angularVelocity = self.speed / turningRadius
            self.direction += math.degrees(angularVelocity * deltaTime)

        self.x += self.speed * \
            math.sin(math.radians(self.direction)) * deltaTime
        self.y -= self.speed * \
            math.cos(math.radians(self.direction)) * deltaTime

        self.rect.x = self.x
        self.rect.y = self.y

    def draw(self, screen):
        rotatedImage = pygame.transform.rotate(self.image, -self.direction)
        imageRect = rotatedImage.get_rect()
        imageRect.center = self.rect.center
        screen.blit(rotatedImage, imageRect)


class CarAgent(Car):
    def __init__(self, x, y, direction, image, model):
        self.model = model
        super().__init__(x, y, direction, image)

    def update(self, deltaTime):
        super().update(deltaTime, 1, 1)