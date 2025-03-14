from config import TRACK_WIDTH, TRACK_HEIGHT, ASSETS_PATH, CAR_WIDTH, CAR_HEIGHT, COLOUR_SCHEME

import pygame
import math
import json

# Loading car images
spawnCarImage = pygame.transform.scale(pygame.image.load(f"{ASSETS_PATH}/Cars/SpawnCar.png"), (CAR_WIDTH, CAR_HEIGHT))

# Initialising fonts
font32 = pygame.font.Font(f"{ASSETS_PATH}/Fonts/font.otf", 32)
font64 = pygame.font.Font(f"{ASSETS_PATH}/Fonts/font.otf", 64)

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

        self.minCurvePoints = 50
        self.pointDensity = 25
        self.finalPointDensity = 5

        self.checkpointFrequency = 15
        self.checkpointOffset = 20

        self.checkpointThickness = 10
        self.checkpointColour = (255, 255, 0)
        self.finishLineColour = (255, 0, 255)

        self.spawnImage = spawnCarImage
        self.spawnRect = self.spawnImage.get_rect()
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

    def getCurve(self, points):
        curve = []
        curvePoints = self.minCurvePoints
        print(points)
        distance = math.dist(points[1], points[2])
        if distance > 1000:
            curvePoints = int(distance / self.pointDensity)

        for index in range(curvePoints + 1):
            t = index / curvePoints

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
            """for i in range(len(self.points) - 3):
                curves.append(self.getCurve(self.points[i:i + 4]))

            for i in range(3):
                curves.append(self.getCurve(self.points[i-3:] + self.points[:i+1]))"""
            
            
            curves.append(self.getCurve([self.points[-1]] + self.points[:3]))
            
            for i in range(len(self.points) - 3):
                curves.append(self.getCurve(self.points[i:i + 4]))
            
            curves.append(self.getCurve(self.points[-3:] + [self.points[0]]))
            curves.append(self.getCurve(self.points[-2:] + self.points[:2]))

        return curves

    def getCheckpoints(self, curves):
        checkpoints = []

        for curve in curves:
            for index in range(len(curve) - self.checkpointOffset * 2 - 1):
                if (index + self.checkpointOffset) % self.checkpointFrequency == 0:
                    point1 = pygame.Vector2(curve[index + self.checkpointOffset])
                    point2 = pygame.Vector2(curve[index + self.checkpointOffset + 1])

                    difference = point2 - point1
                    perpendicularDirection = pygame.Vector2(-difference.y, difference.x).normalize()

                    offset = perpendicularDirection * self.trackWidth
                    checkpoints.append((point1 + offset, point1 - offset))

        """for curve in curves:
            midpoint = len(curve) // 2
            point1 = pygame.Vector2(curve[midpoint])
            point2 = pygame.Vector2(curve[midpoint + 1])

            difference = point2 - point1
            perpendicularDirection = pygame.Vector2(-difference.y, difference.x).normalize()

            offset = perpendicularDirection * self.trackWidth
            checkpoints.append((point1 + offset, point1 - offset))"""

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
            checkpointLabel = font64.render(f"C{index}", True, self.pointLabelColour)
            screen.blit(checkpointLabel, pygame.Vector2(checkpoint[0], checkpoint[1]) + self.pointLabelOffset)

            if index == len(checkpoints) - 2:
                pygame.draw.line(screen, self.finishLineColour, checkpoint[0], checkpoint[1], self.checkpointThickness)
            else:
                pygame.draw.line(screen, self.checkpointColour, checkpoint[0], checkpoint[1], self.checkpointThickness)

        self.spawnRect.center = self.spawnPoint
        screen.blit(self.spawnImage, self.spawnRect)
    
    def initialiseTrack(self):
        curves = self.getCurves()

        self.checkpoints = self.getCheckpoints(curves)
        self.finishLine = self.checkpoints[-2]

        self.pointDensity = self.finalPointDensity
        curves = self.getCurves()

        self.trackSurface = pygame.Surface((TRACK_WIDTH, TRACK_HEIGHT), pygame.SRCALPHA)
        self.drawCircles(self.trackSurface, curves)

        self.mask = pygame.mask.from_surface(self.trackSurface)
        self.mask.invert()

        

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
        
        with open(f"{ASSETS_PATH}/Tracks/{filePath}.json", "w") as file:
            json.dump(output, file)

    def importTrack(self, filePath):
        with open(f"{ASSETS_PATH}/Tracks/{filePath}.json", "r") as file:
            data = json.load(file)

            self.points = data["Points"]
            self.spawnPoint = data["SpawnPoint"]
            self.trackWidth = data["TrackWidth"]
            self.trackColour = data["TrackColour"]
    
    def draw(self, screen, cameraOffset):
        screen.blit(self.trackSurface, (-cameraOffset, (TRACK_WIDTH, TRACK_HEIGHT)))
