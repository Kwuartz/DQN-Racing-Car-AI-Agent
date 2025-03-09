from config import FPS, SCREEN_WIDTH, SCREEN_HEIGHT, CAMERA_SCROLL_SPEED, TRACK_WIDTH, TRACK_HEIGHT, CAR_WIDTH, CAR_HEIGHT, COLOUR_SCHEME, BUTTON_BORDER_THICKNESS, BUTTON_HOVER_THICKNESS, DEFAULT_TRACK_NAME
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
tracksPath = "Assets/Tracks"

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Racing Game")

        self.running = True
        self.clock = pygame.time.Clock()
        self.deltaTime = 1 / FPS

        self.displayMainMenu()

    def displayMainMenu(self):
        playButton = Button(0.1, 0.2, 0.2, 0.1, "Play", font16, COLOUR_SCHEME[0], COLOUR_SCHEME[1],COLOUR_SCHEME[0], BUTTON_BORDER_THICKNESS, BUTTON_HOVER_THICKNESS)
        trackButton = Button(0.1, 0.35, 0.2, 0.1, "Create Track", font16, COLOUR_SCHEME[0], COLOUR_SCHEME[1], COLOUR_SCHEME[0], BUTTON_BORDER_THICKNESS, BUTTON_HOVER_THICKNESS)
        trainButton = Button(0.1, 0.5, 0.2, 0.1, "Train an Agent", font16, COLOUR_SCHEME[0], COLOUR_SCHEME[1], COLOUR_SCHEME[0], BUTTON_BORDER_THICKNESS, BUTTON_HOVER_THICKNESS)
        exitButton = Button(0.1, 0.65, 0.2, 0.1, "Exit", font16, COLOUR_SCHEME[0], COLOUR_SCHEME[1], COLOUR_SCHEME[0], BUTTON_BORDER_THICKNESS, BUTTON_HOVER_THICKNESS)

        buttons = [playButton, trackButton, trainButton, exitButton]

        while self.running:
            hoveredButton = None
            for button in buttons:
                if button.updateHovered(pygame.mouse.get_pos()):
                    hoveredButton = button

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if hoveredButton:
                        if hoveredButton == playButton:
                            trackSelected = self.trackSelection()
                            if trackSelected:
                                self.track.initialiseTrack()
                                self.gameLoop()
                        elif hoveredButton == trackButton:
                            while self.trackSelection(True):
                                self.trackEditor()
                        elif hoveredButton == trainButton:
                            pass
                        elif hoveredButton == exitButton:
                            self.running = False
            
            self.screen.fill((8, 132, 28))

            for button in buttons:
                button.draw(self.screen)

            pygame.display.flip()

            self.deltaTime = self.clock.tick(FPS) / 1000
    
    def trackSelection(self, allowNewTrack=False):
        backButton = Button(0.02, 0.88, 0.1, 0.1, "Back", font16, COLOUR_SCHEME[0], COLOUR_SCHEME[1], COLOUR_SCHEME[0], BUTTON_BORDER_THICKNESS, BUTTON_HOVER_THICKNESS)
        selectButton = Button(0.88, 0.88, 0.1, 0.1, "Select", font16, COLOUR_SCHEME[0], COLOUR_SCHEME[1], COLOUR_SCHEME[0], BUTTON_BORDER_THICKNESS, BUTTON_HOVER_THICKNESS)

        buttons = [backButton, selectButton]

        containerPosition = (0.25, 0.05)
        offScreenPosition = (2, 2)
        buttonLength = 0.48
        buttonHeight = 0.13
        buttonPadding = (0.01, 0.01 * (SCREEN_WIDTH / SCREEN_HEIGHT))
        tracksPerPage = 6
        scrollIndex = 0

        trackButtonsContainer = Container(containerPosition[0], containerPosition[1], buttonLength + buttonPadding[0] * 2, buttonHeight * (tracksPerPage) + buttonPadding[1] * (tracksPerPage + 1), COLOUR_SCHEME[3], COLOUR_SCHEME[0], BUTTON_BORDER_THICKNESS)
        
        trackButtons = []

        if allowNewTrack:
            newTrackButton = Button(0, 0, buttonLength, buttonHeight, "New Track", font16, COLOUR_SCHEME[0], COLOUR_SCHEME[1], COLOUR_SCHEME[0], BUTTON_BORDER_THICKNESS, BUTTON_HOVER_THICKNESS, COLOUR_SCHEME[2])
            trackButtons.append(newTrackButton)

        trackPaths = os.listdir(tracksPath)
        for trackPath in trackPaths:
            strippedPath = trackPath[:-5]
            trackButton = Button(0, 0, buttonLength, buttonHeight, strippedPath, font16, COLOUR_SCHEME[0], COLOUR_SCHEME[1], COLOUR_SCHEME[0], BUTTON_BORDER_THICKNESS, BUTTON_HOVER_THICKNESS, COLOUR_SCHEME[2])
            trackButtons.append(trackButton)

        selectedTrack = 0

        if len(trackButtons) > 0:
            trackButtons[selectedTrack].setSelected(True)

        updateButtons = True
        while self.running:
            hoveredButton = None
            for button in buttons:
                if button.updateHovered(pygame.mouse.get_pos()):
                    hoveredButton = button

            hoveredTrackIndex = None
            for index, button in enumerate(trackButtons):
                if button.updateHovered(pygame.mouse.get_pos()):
                    hoveredTrackIndex = index
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if hoveredTrackIndex is not None:
                        trackButtons[selectedTrack].setSelected(False)
                        selectedTrack = hoveredTrackIndex
                        trackButtons[selectedTrack].setSelected(True)
                    elif hoveredButton == backButton:
                        return False
                    elif hoveredButton == selectButton:
                        if selectedTrack == 0 and allowNewTrack:
                            self.track = Track()
                        else:
                            if allowNewTrack:
                                selectedTrack -= 1

                            self.track = Track(trackPaths[selectedTrack][:-5])

                        return True

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w or event.key == pygame.K_UP:
                        trackButtons[selectedTrack].setSelected(False)
                        selectedTrack = max(selectedTrack - 1, 0)
                        trackButtons[selectedTrack].setSelected(True)
                        
                        if selectedTrack < scrollIndex:
                            scrollIndex -= 1
                            updateButtons = True
                    elif event.key == pygame.K_s or event.key == pygame.K_DOWN:
                        trackButtons[selectedTrack].setSelected(False)
                        selectedTrack = min(selectedTrack + 1, len(trackButtons) - 1)
                        trackButtons[selectedTrack].setSelected(True)

                        if selectedTrack >= scrollIndex + tracksPerPage:
                            scrollIndex += 1  
                            updateButtons = True                  
            
            if updateButtons:
                for index, button in enumerate(trackButtons):
                    if index in range(scrollIndex, scrollIndex + tracksPerPage):
                        relativeIndex = index - scrollIndex
                        button.moveButton(containerPosition[0] + buttonPadding[0], containerPosition[1] + buttonPadding[1] * (relativeIndex + 1) + buttonHeight * relativeIndex)
                    else:
                        button.moveButton(offScreenPosition[0], offScreenPosition[1])

            self.screen.fill((8, 132, 28))

            trackButtonsContainer.draw(self.screen)

            for button in buttons:
                button.draw(self.screen)

            for button in trackButtons:
                button.draw(self.screen)

            pygame.display.flip()

            self.deltaTime = self.clock.tick(FPS) / 1000
            updateButtons = False

    def trackEditor(self):
        trackName = self.track.getFilePath()
        if trackName is None:
            trackName = DEFAULT_TRACK_NAME

        editingTrackName = False

        backButton = Button(0.02, 0.88, 0.1, 0.1, "Back", font16, COLOUR_SCHEME[0], COLOUR_SCHEME[1], COLOUR_SCHEME[0], BUTTON_BORDER_THICKNESS, BUTTON_HOVER_THICKNESS)
        saveButton = Button(0.88, 0.88, 0.1, 0.1, "Save", font16, COLOUR_SCHEME[0], COLOUR_SCHEME[1], COLOUR_SCHEME[0], BUTTON_BORDER_THICKNESS, BUTTON_HOVER_THICKNESS)
        trackNameBox = TextInputBox(0.88, 0.76, 0.1, 0.1, trackName, font16, COLOUR_SCHEME[0], COLOUR_SCHEME[1], COLOUR_SCHEME[0], BUTTON_BORDER_THICKNESS, BUTTON_HOVER_THICKNESS, COLOUR_SCHEME[2])

        elements = [backButton, saveButton, trackNameBox]

        zoom = int(TRACK_WIDTH / SCREEN_WIDTH)
        trackSurface = pygame.Surface((TRACK_WIDTH, TRACK_HEIGHT))
        selectedPoint = None

        editorRunning = True
        while self.running and editorRunning:
            hoveredElement = None
            for element in elements:
                if element.updateHovered(pygame.mouse.get_pos()):
                    hoveredElement = element

            mousePosition = pygame.mouse.get_pos()
            scaledMousePosition = (mousePosition[0] * zoom, mousePosition[1] * zoom)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    editingTrackName = False
                    trackNameBox.setSelected(editingTrackName)

                    if event.button == 1:
                        if hoveredElement:
                            if hoveredElement == backButton:
                                editorRunning = False
                            elif hoveredElement == saveButton:
                                if selectedPoint is None:
                                    trackName = trackNameBox.getText()
                                    if trackName == "":
                                        trackName = DEFAULT_TRACK_NAME

                                    self.track.exportTrack(trackName)
                                    editorRunning = False

                            elif hoveredElement == trackNameBox:
                                editingTrackName = True
                                trackNameBox.setSelected(editingTrackName)

                        elif selectedPoint is not None:
                            selectedPoint = None
                        else:
                            if (hoveredPoint := self.track.getHoveredPoint(scaledMousePosition)) is not None:
                                selectedPoint = hoveredPoint
                            else:
                                self.track.addPoint(scaledMousePosition)

                    elif event.button == 3:
                        if (hoveredPoint := self.track.getHoveredPoint(scaledMousePosition)) is not None:
                            self.track.removePoint(hoveredPoint)
                        else:
                            self.track.removePoint()

                elif event.type == pygame.KEYDOWN and editingTrackName:
                    trackNameBox.update(event)

            if selectedPoint is not None:
                self.track.movePoint(selectedPoint, scaledMousePosition)
            
            trackSurface.fill((8, 132, 28))
            self.track.drawEditor(trackSurface)

            scaledTrackSurface = pygame.transform.scale(trackSurface, (SCREEN_WIDTH, SCREEN_HEIGHT))
            self.screen.blit(scaledTrackSurface, (0, 0))

            for element in elements:
                element.draw(self.screen)

            pygame.display.flip()

            self.deltaTime = self.clock.tick(FPS) / 1000

    def training(self):
        pass

    def gameLoop(self):
        spawnPoint = self.track.getSpawnPoint()

        self.playerCar = Car(spawnPoint[0], spawnPoint[1], 0, blueCarImage)
        self.agentCar = CarAgent(spawnPoint[0], spawnPoint[1], 0, redCarImage)

        self.cameraOffset = pygame.Vector2(0, 0)

        gameRunning = True
        while self.running and gameRunning:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            acceleration = 0
            turnDirection = 0

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                acceleration += 1
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                acceleration -= 1
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                turnDirection += 1
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
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

        self.steerSpeed = 90
        self.steerCenterSpeed = 90

        self.x = x
        self.y = y

        self.speed = 0
        self.wheelDirection = direction
        self.direction = direction

        self.image = image
        self.rotatedImage = pygame.transform.rotate(self.image, -self.direction)
        self.mask = pygame.mask.from_surface(self.image)
        self.maskOffset = (0, 0)

        self.rect = image.get_rect()
        self.imageRect = self.rect
        self.rect.center = (x, y)

    def update(self, deltaTime, acceleration, steerDirection, track):
        self.track = track
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
            turningRadius = (self.rect.height * 4) / \
                math.tan(math.radians(self.wheelDirection))
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
        else:
            self.speed = 0
        
        overlapY = track.getOverlap(self.imageRect.x, self.imageRect.y - yChange, self.mask)
        if not overlapY:
            self.y -= yChange
            self.rect.y = self.y
            self.imageRect.center = self.rect.center
        else:
            self.speed = 0

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
                screen.set_at((round(position.x - cameraOffset[0]), round(position.y - cameraOffset[1])), (255, 0, 0))

                if self.track.checkCollideAtPoint(position):
                    sensorDistance = distance
                    break
            distances.append(sensorDistance)
        
        return distances

    def draw(self, screen, cameraOffset):
        screen.blit(self.rotatedImage, (self.imageRect.x - cameraOffset[0], self.imageRect.y - cameraOffset[1]))
        self.getDistances(screen, cameraOffset)

class CarAgent(Car):
    def __init__(self, x, y, direction, image, model=False):
        super().__init__(x, y, direction, image)
        
        self.sensors= [
            -180,
            -135,
            -45,
            0,
        ]

        self.model = False

    def update(self, deltaTime, track):
        super().update(deltaTime, random.randint(-1, 1), 1, track)
    
    def getSensors(self, sensorCount):
        frontOffset = (self.rect.width, 0)
        sensors = []

        for index in range(sensorCount):
            offset = pygame.Vector2(0, self.rect.height / sensorCount) + frontOffset
            angle = self.fov / sensorCount
            sensors.append((offset, angle))

        return sensors

    def getDistances(self, track, maxDistance=200):
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
            
    def draw(self, screen, cameraOffset):
        screen.blit(self.rotatedImage, (self.imageRect.x - cameraOffset[0], self.imageRect.y - cameraOffset[1]))
    
class Track:
    def __init__(self, filePath=None):
        self.filePath = filePath

        if self.filePath:
            self.importTrack(filePath)
        else:
            self.points = []
            self.spawnPoint = (TRACK_WIDTH / 2, TRACK_HEIGHT / 2)

            self.trackWidth = 250
            self.trackColour = (50, 50, 50)

        self.checkpoints = None
        self.finishLine = None

        self.checkpointThickness = 10
        self.checkpointColour = (255, 255, 0)
        self.finishLineColour = (255, 0, 255)

        self.spawnImage = blueCarImage
        self.spawnRect = blueCarImage.get_rect()
        self.spawnRect.center = self.spawnPoint

        self.pointRadius = 15
        self.pointColour = (255, 0, 0)

        self.pointLabelOffset = pygame.Vector2(10, 10)
        self.pointLabelColour = COLOUR_SCHEME[1]

        self.lineThickness = 10
        self.lineColour = COLOUR_SCHEME[1]
        self.connectingLineColour = COLOUR_SCHEME[0]

    def getSpawnPoint(self):
        return self.spawnPoint

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

    def getCurve(self, points, n=50):
        curve = []

        distance = math.dist(points[1], points[2])
        if distance > 1000:
            n = int(distance / 25)

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

    def getCheckpoints(self, curves):
        checkpoints = []

        for curve in curves:
            midpoint = len(curve) // 2
            point1 = pygame.Vector2(curve[midpoint])
            point2 = pygame.Vector2(curve[midpoint + 1])

            difference = point2 - point1
            perpendicularDirection = pygame.Vector2(-difference.y, difference.x).normalize()

            offset = perpendicularDirection * self.trackWidth
            checkpoints.append((point1 + offset, point1 - offset))

        return checkpoints

    def drawCircles(self, screen, curves):
        for curve in curves:
            for point in curve:
                pygame.draw.circle(screen, self.trackColour, point, self.trackWidth)

    def drawEditor(self, screen):
        curves = self.getCurves()

        self.drawCircles(screen, curves)

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

        checkpoints = self.getCheckpoints(curves)
        for index, checkpoint in enumerate(checkpoints):
            if index == len(checkpoints) - 2:
                pygame.draw.line(screen, self.finishLineColour, checkpoint[0], checkpoint[1], self.checkpointThickness)
            else:
                pygame.draw.line(screen, self.checkpointColour, checkpoint[0], checkpoint[1], self.checkpointThickness)

        self.spawnRect.center = self.spawnPoint
        screen.blit(self.spawnImage, self.spawnRect)
    
    def initialiseTrack(self):
        curves = self.getCurves()
        self.trackSurface = pygame.Surface((TRACK_WIDTH, TRACK_HEIGHT), pygame.SRCALPHA)
        self.drawCircles(self.trackSurface, curves)

        self.mask = pygame.mask.from_surface(self.trackSurface)
        self.mask.invert()

        self.checkpoints = self.getCheckpoints(curves)
        self.finishLine = self.checkpoints[-2]

    def getOverlap(self, x, y, mask):
        overlap = self.mask.overlap(mask, (x, y))
        return overlap

    def checkCollideAtPoint(self, position):
        return self.mask.get_at(position)

    def getFilePath(self):
        return self.filePath

    def exportTrack(self, filePath):
        output = {
            "Points": self.points,
            "SpawnPoint": self.spawnPoint,
            "TrackWidth": self.trackWidth,
            "TrackColour": self.trackColour
        }
        
        with open(f"{tracksPath}/{filePath}.json", "w") as file:
            json.dump(output, file)

    def importTrack(self, filePath):
        with open(f"{tracksPath}/{filePath}.json", "r") as file:
            data = json.load(file)

            self.points = data["Points"]
            self.spawnPoint = data["SpawnPoint"]
            self.trackWidth = data["TrackWidth"]
            self.trackColour = data["TrackColour"]
    
    def draw(self, screen, cameraOffset):
        screen.blit(self.trackSurface, (-cameraOffset, (TRACK_WIDTH, TRACK_HEIGHT)))

class Container:
    def __init__(self, x, y, width, height, backgroundColour, borderColour, borderThickness):
        self.rect = pygame.Rect(SCREEN_WIDTH * x, SCREEN_HEIGHT * y, SCREEN_WIDTH * width, SCREEN_HEIGHT * height)

        self.backgroundColour = backgroundColour
        
        self.borderColour = borderColour
        self.borderThickness = borderThickness

    def draw(self, screen):
        pygame.draw.rect(screen, self.backgroundColour, self.rect)
        pygame.draw.rect(screen, self.borderColour, self.rect, self.borderThickness)

class Button:
    def __init__(self, x, y, width, height, text, font, textColour, backgroundColour, borderColour, borderThickness, hoverBorderThickness=0, selectedBackgroundColour=(0,0,0)):
        self.font = font
        self.textColour = textColour

        self.text = font.render(text, True, textColour)
        self.rect = pygame.Rect(SCREEN_WIDTH * x, SCREEN_HEIGHT * y, SCREEN_WIDTH * width, SCREEN_HEIGHT * height)

        self.backgroundColour = backgroundColour
        self.normalBackgroundColour = backgroundColour
        self.selectedBackgroundColour = selectedBackgroundColour

        self.borderColour = borderColour
        self.borderThickness = borderThickness
        self.normalBorderThickness = borderThickness
        self.hoverBorderThickness = hoverBorderThickness

    def draw(self, screen):
        pygame.draw.rect(screen, self.backgroundColour, self.rect)
        pygame.draw.rect(screen, self.borderColour, self.rect, self.borderThickness)
        screen.blit(self.text, self.text.get_rect(center=self.rect.center))

    def moveButton(self, x, y):
        self.rect.x = SCREEN_WIDTH * x
        self.rect.y = SCREEN_HEIGHT * y

    def updateHovered(self, mousePosition):
        if self.rect.collidepoint(mousePosition):
            self.borderThickness = self.hoverBorderThickness
            return True
        else:
            self.borderThickness = self.normalBorderThickness

    def setSelected(self, selected):
        if selected:
            self.backgroundColour = self.selectedBackgroundColour
        else:
            self.backgroundColour = self.normalBackgroundColour

class TextInputBox(Button):
    def __init__(self, x, y, width, height, text, font, textColour, backgroundColour, borderColour, borderThickness, hoverBorderThickness=0, selectedBackgroundColour=(0,0,0)):
        self.textContent = text
        super().__init__(x, y, width, height, text, font, textColour, backgroundColour, borderColour, borderThickness, hoverBorderThickness, selectedBackgroundColour)

    def update(self, event):
        if event.key == pygame.K_BACKSPACE:
            self.textContent = self.textContent[:-1]
        elif event.unicode.isalnum():
            self.textContent += event.unicode

        self.text = self.font.render(self.textContent, True, self.textColour)

    def getText(self):
        return self.textContent

if __name__ == "__main__":
    Game()