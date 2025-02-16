from config import FPS, SCREEN_WIDTH, SCREEN_HEIGHT
import pygame
import random
import math

class Game:
    def __init__(self):
        pygame.init()
        
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Racing Game")
        
        self.font16 = pygame.font.Font("assets/Fonts/font16.otf", 16)
        self.blueCarImage = pygame.image.load("Assets/Cars/BlueCar.png")
        self.redCarImage = pygame.image.load("Assets/Cars/RedCar.png")

        self.training = False
        self.running = True

        self.displayMainMenu()

    

    def displayMainMenu(self):
        self.clock = pygame.time.Clock()
        self.deltaTime = 1 / FPS
        self.playButton = Button(0.1, 0.2, 0.2, 0.1, "Play", self.font16, (255, 255, 255), (255, 255, 255), 5)
        self.trainButton = Button(0.1, 0.35, 0.2, 0.1, "Train", self.font16, (255, 255, 255), (255, 255, 255), 5)
        self.exitButton = Button(0.1, 0.5, 0.2, 0.1, "Exit", self.font16, (255, 255, 255), (255, 255, 255), 5)

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.playButton.checkHovered(pygame.mouse.get_pos()):
                        self.gameLoop()
                    elif self.trainButton.checkHovered(pygame.mouse.get_pos()):
                        self.training = True
                        pass
                    elif self.exitButton.checkHovered(pygame.mouse.get_pos()):
                        self.running = False
            
            self.playButton.draw(self.screen)
            self.trainButton.draw(self.screen)
            self.exitButton.draw(self.screen)

            pygame.display.flip()

            self.deltaTime = self.clock.tick(FPS) / 1000
        
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

        self.agentCar.update(self.deltaTime)

    def drawGame(self):
        self.screen.fill((0, 0, 0))

        if not self.training:
            self.playerCar.draw(self.screen)
        
        self.agentCar.draw(self.screen)

        pygame.display.flip()

    def gameLoop(self):
        gameRunning = True

        if not self.training:
            self.playerCar = Car(300, 300, 0, self.blueCarImage)

        self.agentCar = CarAgent(300, 300, 0, self.redCarImage)

        while self.running and gameRunning:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.update()
            self.drawGame()
            
            self.deltaTime = self.clock.tick(FPS) / 1000

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
    def __init__(self, x, y, direction, image, model=False):
        self.model = model
        super().__init__(x, y, direction, image)

    def update(self, deltaTime):
        super().update(deltaTime, random.randint(-1, 1), 1)

class Button:
    def __init__(self, x, y, width, height, text, font, textColor, bgColour, rectThickness = 0):
        self.text = font.render(text, True, textColor)
        self.rect = pygame.Rect(0, 0, SCREEN_WIDTH * width, SCREEN_HEIGHT * height)
        self.rect.center = (SCREEN_WIDTH * x, SCREEN_HEIGHT * y)
        self.rectThickness = rectThickness
        self.bgColour = bgColour

    def draw(self, screen):
        pygame.draw.rect(screen, self.bgColour, self.rect, self.rectThickness)
        screen.blit(self.text, self.text.get_rect(center=self.rect.center))

    def checkHovered(self, mousePosition):
        return self.rect.collidepoint(mousePosition)

if __name__ == "__main__":
    Game()