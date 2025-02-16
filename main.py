from config import FPS, SCREEN_WIDTH, SCREEN_HEIGHT, TRACK, CAMERA_SCROLL_SPEED
import pygame
import random
import math

class Game:
    def __init__(self):
        pygame.init()
        
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Racing Game")
        
        self.font16 = pygame.font.Font("Assets/Fonts/font16.otf", 16)
        self.blueCarImage = pygame.image.load("Assets/Cars/BlueCar.png")
        self.redCarImage = pygame.image.load("Assets/Cars/RedCar.png")
        self.trackImage = pygame.image.load(f"Assets/Tracks/{TRACK}.png").convert_alpha()

        self.track = Track(self.trackImage)
        self.track.invertMask()
        
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
            
            self.screen.fill((8, 132, 28))

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

            self.playerCar.update(self.deltaTime, acceleration, turnDirection, self.track)

        self.agentCar.update(self.deltaTime, self.track)

    def drawGame(self):
        self.screen.fill((8, 132, 28))

        if self.training:
            self.cameraOffset = self.agentCar.getCameraOffset(self.cameraOffset)
        else:
            self.cameraOffset = self.playerCar.getCameraOffset(self.cameraOffset)

        self.track.draw(self.screen, self.cameraOffset)

        self.agentCar.draw(self.screen, self.cameraOffset)

        if not self.training:
            self.playerCar.draw(self.screen, self.cameraOffset)

        pygame.display.flip()

    def gameLoop(self):
        gameRunning = True
        self.cameraOffset = pygame.Vector2(0, 0)

        if not self.training:
            self.playerCar = Car(1000, 1000, 0, self.blueCarImage)

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
        self.maxSpeed = 700
        self.acceleration = 200

        self.friction = 0.5
        self.brakeStrength = 3
        self.stoppingOffset = 50

        self.steerSpeed = 90
        self.steerCenterSpeed = 90

        self.width = 80
        self.height = 106

        self.x = x
        self.y = y

        self.speed = 0
        self.wheelDirection = direction
        self.direction = direction

        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.image = pygame.transform.scale(image, (self.width, self.height))
        self.mask = pygame.mask.from_surface(self.image)

    def update(self, deltaTime, acceleration, steerDirection, track):
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
            self.wheelDirection = max(-35, min(self.wheelDirection, 35))
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
        
        xChange = self.speed * \
            math.sin(math.radians(self.direction)) * deltaTime
        yChange = self.speed * \
            math.cos(math.radians(self.direction)) * deltaTime

        overlapX = track.getOverlap(self.x + xChange, self.y, self.mask)
        if not overlapX:
            self.x += xChange
        else:
            self.speed -= self.speed * abs(math.sin(math.radians(self.direction))) * (10/FPS)
        
        overlapY = track.getOverlap(self.x, self.y - yChange, self.mask)
        if not overlapY:
            self.y -= yChange
        else:
            self.speed -= self.speed * abs(math.cos(math.radians(self.direction))) * (10/FPS)

        self.rect.x = self.x
        self.rect.y = self.y

    def getCameraOffset(self, cameraOffset):
        xTrueOffset = self.x - SCREEN_WIDTH / 2
        yTrueOffset = self.y - SCREEN_HEIGHT / 2

        xOffset = (xTrueOffset - cameraOffset[0]) * CAMERA_SCROLL_SPEED
        yOffset = (yTrueOffset - cameraOffset[1]) * CAMERA_SCROLL_SPEED

        return cameraOffset + pygame.Vector2(xOffset, yOffset)

    def draw(self, screen, cameraOffset):
        rotatedImage = pygame.transform.rotate(self.image, -self.direction)
        imageRect = rotatedImage.get_rect()
        imageRect.center = self.rect.center - cameraOffset
        screen.blit(rotatedImage, imageRect)

class CarAgent(Car):
    def __init__(self, x, y, direction, image, model=False):
        self.model = model
        super().__init__(x, y, direction, image)

    def update(self, deltaTime, track):
        super().update(deltaTime, random.randint(-1, 1), 1, track)

class Track:
    def __init__(self, image):
        self.image = pygame.transform.scale(image, (image.get_width() * 10, image.get_height() * 10))
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()

    def invertMask(self):
        maskSize = self.mask.get_size()

        for x in range(maskSize[0]):
            for y in range(maskSize[1]):
                if self.mask.get_at((x, y)):
                    self.mask.set_at((x, y), 0)
                else:
                    self.mask.set_at((x, y), 1)
        
    def getOverlap(self, x, y, mask):
        offset = (int(x - self.rect.x), int(y - self.rect.y))
        overlap = self.mask.overlap(mask, offset)
        return overlap

    def draw(self, screen, cameraOffset):
        screen.blit(self.image, (self.rect.topleft - cameraOffset, (self.rect.width, self.rect.height)))

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