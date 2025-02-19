from config import FPS, SCREEN_WIDTH, SCREEN_HEIGHT, TRACK, CAMERA_SCROLL_SPEED, TRACK_WIDTH, TRACK_HEIGHT, CAR_WIDTH, CAR_HEIGHT, ASSETS_PATH
import pygame
import random
import json
import math
import os

pygame.init()

font16 = pygame.font.Font("Assets/Fonts/font.otf", 16)
font32 = pygame.font.Font("Assets/Fonts/font.otf", 32)
font64 = pygame.font.Font("Assets/Fonts/font.otf", 64)

blueCarImage = pygame.transform.scale(pygame.image.load("Assets/Cars/BlueCar.png"), (CAR_WIDTH, CAR_HEIGHT))
redCarImage = pygame.transform.scale(pygame.image.load("Assets/Cars/RedCar.png"), (CAR_WIDTH, CAR_HEIGHT))

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Racing Game")

        self.running = True
        self.clock = pygame.time.Clock()
        self.deltaTime = 1 / FPS

        self.displayMainMenu()

    def displayMainMenu(self):
        self.playButton = Button(0.1, 0.2, 0.2, 0.1, "Play", font16, (255, 255, 255), (0, 0, 0),(255, 255, 255), 5, 7)
        self.trackButton = Button(0.1, 0.35, 0.2, 0.1, "Create Track", font16, (255, 255, 255), (0, 0, 0), (255, 255, 255), 5, 7)
        self.trainButton = Button(0.1, 0.5, 0.2, 0.1, "Train an Agent", font16, (255, 255, 255), (0, 0, 0), (255, 255, 255), 5, 7)
        self.exitButton = Button(0.1, 0.65, 0.2, 0.1, "Exit", font16, (255, 255, 255), (0, 0, 0), (255, 255, 255), 5, 7)

        buttons = [self.playButton, self.trackButton, self.trainButton, self.exitButton]

        while self.running:
            hoveredButton = None
            for button in buttons:
                button.setHovered(False)

                if button.checkHovered(pygame.mouse.get_pos()):
                    button.setHovered(True)
                    hoveredButton = button

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if hoveredButton:
                        if hoveredButton == self.playButton:
                            self.trackSelection()
                            self.gameLoop()

                        elif hoveredButton == self.trackButton:#
                            self.trackSelection()
                            self.trackEditor()

                        elif hoveredButton == self.trainButton:
                            pass

                        elif hoveredButton == self.exitButton:
                            self.running = False
            
            self.screen.fill((8, 132, 28))

            self.playButton.draw(self.screen)
            self.trackButton.draw(self.screen)
            self.trainButton.draw(self.screen)
            self.exitButton.draw(self.screen)

            pygame.display.flip()

            self.deltaTime = self.clock.tick(FPS) / 1000
    
    def trackSelection(self):
        tracksPath = f"{ASSETS_PATH}/Tracks"
        tracksPaths = os.listdir(tracksPath)
        
        trackButtons = []
        for trackPath in tracksPaths:
            strippedPath = trackPath[:-5]
            trackButton = Button(0.1, 0.2, 0.2, 0.1, strippedPath, font16, (255, 255, 255), (0, 0, 0),(255, 255, 255), 5, 7)
            trackButtons.append(trackButton)

        selectedTrack = 0
        trackButtons[selectedTrack].setHovered(True)

        selectedTrack = True
        while self.running and selectingTrack:
            hoveredButtonIndex = None
            for index, button in enumerate(trackButtons):
                button.setHovered(False)

                if button.checkHovered(pygame.mouse.get_pos()):
                    button.setHovered(True)
                    hoveredButtonIndex = index

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if hoveredButtonIndex is not None:
                        selectedTrack = hoveredButtonIndex
            
            self.screen.fill((8, 132, 28))

            for button in trackButtons:
                button.draw(self.screen)

            pygame.display.flip()

            self.deltaTime = self.clock.tick(FPS) / 1000

    def trackEditor(self):
        zoom = int(TRACK_WIDTH / SCREEN_WIDTH)
        trackSurface = pygame.Surface((TRACK_WIDTH, TRACK_HEIGHT))
        selectedPoint = None

        editorRunning = True
        while self.running and editorRunning:
            updateGraphics = False

            mousePosition = pygame.mouse.get_pos()
            scaledMousePosition = (mousePosition[0] * zoom, mousePosition[1] * zoom)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if selectedPoint is not None:
                            selectedPoint = None
                        else:
                            if (hoveredPoint := self.track.getHoveredPoint(scaledMousePosition)) is not None:
                                selectedPoint = hoveredPoint
                            else:
                                self.track.addPoint(scaledMousePosition)
                                updateGraphics = True

                    elif event.button == 3:
                        if (hoveredPoint := self.track.getHoveredPoint(scaledMousePosition)) is not None:
                            self.track.removePoint(hoveredPoint)
                        else:
                            self.track.removePoint()

                        updateGraphics = True

            if selectedPoint is not None:
                self.track.movePoint(selectedPoint, scaledMousePosition)
                updateGraphics = True
            
            if updateGraphics:
                trackSurface.fill((8, 132, 28))
                self.track.drawEditor(trackSurface)

                scaledTrackSurface = pygame.transform.scale(trackSurface, (SCREEN_WIDTH, SCREEN_HEIGHT))
                self.screen.blit(scaledTrackSurface, (0, 0))

                pygame.display.flip()

            self.deltaTime = self.clock.tick(FPS) / 1000

    def training(self):
        pass

    def gameLoop(self):
        self.playerCar = Car(1000, 1000, 0, blueCarImage)
        self.agentCar = CarAgent(300, 300, 0, redCarImage)

        self.cameraOffset = pygame.Vector2(0, 0)

        gameRunning = True
        while self.running and gameRunning:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

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

            self.cameraOffset = self.playerCar.getCameraOffset(self.cameraOffset)
            
            self.screen.fill((8, 132, 28))
            self.track.draw(self.screen, self.cameraOffset)

            self.playerCar.draw(self.screen, self.cameraOffset)
            self.agentCar.draw(self.screen, self.cameraOffset)

            pygame.display.flip()

            self.deltaTime = self.clock.tick(FPS) / 1000

class Car:
    def __init__(self, x, y, direction, image):
        self.maxSpeed = 800
        self.acceleration = 300

        self.friction = 0.5
        self.brakeStrength = 3
        self.stoppingOffset = 50

        self.steerSpeed = 60
        self.steerCenterSpeed = 90

        self.x = x
        self.y = y

        self.speed = 0
        self.wheelDirection = direction
        self.direction = direction

        self.image = image
        self.mask = pygame.mask.from_surface(self.image)

        self.rect = image.get_rect()
        self.rect.center = (x, y)

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
            self.wheelDirection = max(-45, min(self.wheelDirection, 45))
        else:
            if self.wheelDirection > 0:
                self.wheelDirection = max(
                    self.wheelDirection - self.steerCenterSpeed * deltaTime, 0)
            elif self.wheelDirection < 0:
                self.wheelDirection = min(
                    self.wheelDirection + self.steerCenterSpeed * deltaTime, 0)

        if self.speed != 0 and self.wheelDirection != 0:
            turningRadius = (self.height * 4) / \
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
    def __init__(self, filePath = None):
        if filePath:
            self.importTrack(filePath)
        else:
            self.points = []
            self.spawnPoint = pygame.Vector2(TRACK_WIDTH / 2, TRACK_HEIGHT / 2)

            self.trackWidth = 150
            self.trackColour = (50, 50, 50)

        self.spawnImage = blueCarImage
        self.spawnRect = blueCarImage.get_rect()

        self.pointRadius = 15
        self.pointColour = (255, 0, 0)

        self.pointLabelOffset = pygame.Vector2(10, 10)
        self.pointLabelColour = (0, 0, 0)

        self.lineThickness = 10
        self.lineColour = (0, 0, 0)
        self.connectingLineColour = (255, 255, 255)

    def addPoint(self, pos):
        self.points.append(pos)

    def movePoint(self, index, newPosition):
        if index == "spawn":
            self.spawnPoint = newPosition
        else:
            self.points[index] = newPosition

    def removePoint(self, index=-1):
        if len(self.points) > 0 and index != "spawn":
            self.points.pop(index)

    def getPoints(self):
        return self.points

    def getHoveredPoint(self, mousePosition):
        for index, point in enumerate(self.points):
            if math.dist(mousePosition, point) < self.pointRadius:
                return index

        if self.spawnRect.collidepoint(mousePosition):
            return "spawn"

    def getLines(self):
        lines = []

        for i in range(len(self.points) - 2):
            lines += self.getCurve(self.points[i:i+3])

        return lines

    def getCurve(self, points, n=20):
        curve = []

        distance = math.dist(points[1], points[2])
        if distance > 1000:
            n = int(distance / 50)

        for i in range(n + 1):
            t = i / n

            x = 0.5 * (
                (2 * points[1][0]) +
                (-points[0][0] + points[2][0]) * t +
                (2 * points[0][0] - 5 * points[1][0] + 4 * points[2][0] - points[3][0]) * t**2 +
                (-points[0][0] + 3 * points[1][0] - 3 * points[2][0] + points[3][0]) * t**3
            )

            y = 0.5 * (
                (2 * points[1][1]) +
                (-points[0][1] + points[2][1]) * t +
                (2 * points[0][1] - 5 * points[1][1] + 4 * points[2][1] - points[3][1]) * t**2 +
                (-points[0][1] + 3 * points[1][1] - 3 * points[2][1] + points[3][1]) * t**3
            )

            curve.append((x, y))

        return curve

    def getCurves(self):
        curves = []

        if len(self.points) > 3:
            for i in range(len(self.points) - 3):
                curves.append(self.getCurve(self.points[i:i + 4]))

            for i in range(3):
                curves.append(self.getCurve(self.points[i-3:] + self.points[:i+1]))

        return curves

    def draw(self, screen):
        curves = self.getCurves()
        
        for curve in curves:
            for point in curve:
                pygame.draw.circle(screen, self.trackColour, point, self.trackWidth)

    def drawEditor(self, screen):
        curves = self.getCurves()
        
        for curve in curves:
            for point in curve:
                pygame.draw.circle(screen, self.trackColour, point, self.trackWidth)

        for index, curve in enumerate(curves):
            for point in curve:
                if index == len(curves) - 2:
                    pygame.draw.lines(screen, self.connectingLineColour, False, curve, self.lineThickness)
                else:
                    pygame.draw.lines(screen, self.lineColour, False, curve, self.lineThickness)
        
        for index, point in enumerate(self.points):
            pointLabel = font64.render(f"P{index}", True, self.pointLabelColour)
            
            pygame.draw.circle(screen, self.pointColour, point, self.pointRadius)
            screen.blit(pointLabel, point + self.pointLabelOffset)

        self.spawnRect.center = self.spawnPoint
        screen.blit(self.spawnImage, self.spawnRect)
    
    def createSurfaceAndMask(self):
        self.trackSurface = pygame.Surface((TRACK_WIDTH, TRACK_HEIGHT))
        self.draw(trackSurface)

        self.mask = pygame.mask.from_surface(trackSurface)
        self.mask.invert()

    def getOverlap(self, x, y, mask):
        offset = (int(x - self.rect.x), int(y - self.rect.y))
        overlap = self.mask.overlap(mask, offset)
        return overlap

    def exportTrack(self, filePath):
        output = {
            "Points": self.points,
            "SpawnPoint": self.spawnPoint,
            "TrackWidth": self.trackWidth,
            "TrackColour": self.trackColour
        }
        
        with open(f"{filePath}.json", "w") as file:
            json.dump(output, file)

    def importTrack(self, filePath):
        with open(f"{filePath}.json", "r") as file:
            data = json.load(file)

            self.points = data["Points"]
            self.spawnPoint = data["SpawnPoint"]
            self.trackWidth = data["TrackWidth"]
            self.trackColour = data["TrackColour"]

    def getOverlap(self, x, y, mask):
        offset = (int(x - self.rect.x), int(y - self.rect.y))
        overlap = self.mask.overlap(mask, offset)
        return overlap

    def draw(self, screen, cameraOffset):
        screen.blit(self.image, (self.rect.topleft - cameraOffset, (self.rect.width, self.rect.height)))

class ImageTrack:
    def __init__(self, image):
        self.image = pygame.transform.scale(image, (image.get_width() * 10, image.get_height() * 10))
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()

    def invertMask(self):
        self.mask.invert()

class Button:
    def __init__(self, x, y, width, height, text, font, textColour, backgroundColour, borderColour=(0,0,0), rectThickness=0, hoverThickness=0):
        self.text = font.render(text, True, textColour)
        self.rect = pygame.Rect(SCREEN_WIDTH * x, SCREEN_HEIGHT * y, SCREEN_WIDTH * width, SCREEN_HEIGHT * height)

        self.backgroundColour = backgroundColour
        self.borderColour = borderColour

        self.rectThickness = rectThickness
        self.normalThickness = rectThickness
        self.hoverThickness = hoverThickness

    def draw(self, screen):
        pygame.draw.rect(screen, self.backgroundColour, self.rect)
        pygame.draw.rect(screen, self.borderColour, self.rect, self.rectThickness)
        screen.blit(self.text, self.text.get_rect(center=self.rect.center))

    def moveButton(self, x, y):
        self.rect.x = x
        self.rect.y = y

    def checkHovered(self, mousePosition):
        return self.rect.collidepoint(mousePosition)

    def setHovered(self, hovered):
        if hovered:
            self.rectThickness = self.hoverThickness
        else:
            self.rectThickness = self.normalThickness

if __name__ == "__main__":
    Game()