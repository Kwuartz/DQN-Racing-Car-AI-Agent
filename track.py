from config import TRACK_WIDTH, TRACK_HEIGHT, TRACKS_PATH, COLOUR_SCHEME, CHECKPOINT_FREQUENCY, FONT_64

import pygame
import math
import json

class Track:
    def __init__(self, filePath=None):
        self.filePath = filePath

        if self.filePath:
            self.importTrack(filePath)
        else:
            self.points = []

            self.trackWidth = 250
            self.trackColour = (50, 50, 50)

        self.curves = None
        self.checkpoints = None

        self.minCurvePoints = 50
        self.pointFrequency = 20
        self.finalPointFrequency = 5

        self.checkpointOffset = 7
        
        self.finishLineThickness = 20
        self.finishLineColour = (0, 0, 0)

        self.checkpointThickness = 10
        self.checkpointColour = (255, 255, 0)

        self.pointRadius = 15
        self.pointColour = (255, 0, 0)

        self.pointLabelOffset = pygame.Vector2(10, 10)
        self.pointLabelColour = COLOUR_SCHEME[1]

        self.lineThickness = 10
        self.lineColour = COLOUR_SCHEME[1]

    def addPoint(self, position):
        self.points.append(position)

    def movePoint(self, index, newPosition):
        self.points[index] = newPosition

    def removePoint(self, index=-1):
        if len(self.points) > 0:
            self.points.pop(index)

    def getPoints(self):
        return self.points

    def getHoveredPoint(self, mousePosition):
        for index, point in enumerate(self.points):
            if math.dist(mousePosition, point) < self.pointRadius:
                return index

    def getLines(self):
        lines = []

        for i in range(len(self.points) - 2):
            lines += self.getCurve(self.points[i:i+3])

        return lines

    def getCurve(self, points):
        curve = []
        curvePoints = self.minCurvePoints

        distance = math.dist(points[1], points[2])
        if distance > 1000:
            curvePoints = int(distance / self.pointFrequency)

        for index in range(curvePoints + 1):
            t = index / curvePoints

            x = 0.5 * (
                (2 * points[1].x) +
                (-points[0].x + points[2].x) * t +
                (2 * points[0].x - 5 * points[1].x + 4 * points[2].x - points[3].x) * t**2 +
                (-points[0].x + 3 * points[1].x - 3 * points[2].x + points[3].x) * t**3
            )

            y = 0.5 * (
                (2 * points[1].y) +
                (-points[0].y + points[2].y) * t +
                (2 * points[0].y - 5 * points[1].y + 4 * points[2].y - points[3].y) * t**2 +
                (-points[0].y + 3 * points[1].y - 3 * points[2].y + points[3].y) * t**3
            )

            curve.append(pygame.Vector2(x, y))

        return curve

    def getCurves(self):
        curves = []

        if len(self.points) > 3:
            curves.append(self.getCurve([self.points[-1]] + self.points[:3]))
            
            for i in range(len(self.points) - 3):
                curves.append(self.getCurve(self.points[i:i + 4]))
            
            curves.append(self.getCurve(self.points[-3:] + [self.points[0]]))
            curves.append(self.getCurve(self.points[-2:] + self.points[:2]))

        return curves

    def getCheckpoints(self, curves):
        checkpoints = []
        # restor midpoint code so if there is not enough points in a curve we can add a checkpoint at midpoint of a curve
        for curve in curves:
            for index in range(len(curve) - self.checkpointOffset * 2 - 1):
                if (index + self.checkpointOffset) % CHECKPOINT_FREQUENCY == 0:
                    point1 = pygame.Vector2(curve[index + self.checkpointOffset])
                    point2 = pygame.Vector2(curve[index + self.checkpointOffset + 1])

                    difference = point2 - point1
                    perpendicularDirection = pygame.Vector2(-difference.y, difference.x).normalize()

                    offset = perpendicularDirection * self.trackWidth
                    checkpoints.append((point1 + offset, point1 - offset))

        return checkpoints

    def drawCircles(self, screen, curves):
        for curve in curves:
            for point in curve:
                pygame.draw.circle(screen, self.trackColour, point, self.trackWidth)

    def drawEditor(self, screen, drawCheckpoints=False):
        curves = self.getCurves()
        self.drawCircles(screen, curves)

        for index, curve in enumerate(curves):
            for point in curve:
                pygame.draw.lines(screen, self.lineColour, False, curve, self.lineThickness)
        
        for index, point in enumerate(self.points):
            pointLabel = FONT_64.render(f"P{index}", True, self.pointLabelColour)
            
            pygame.draw.circle(screen, self.pointColour, point, self.pointRadius)
            screen.blit(pointLabel, point + self.pointLabelOffset)
        
        checkpoints = self.getCheckpoints(curves)
    
        if drawCheckpoints:
            for index, checkpoint in enumerate(checkpoints):
                checkpointLabel = FONT_64.render(f"C{index}", True, self.pointLabelColour)
                screen.blit(checkpointLabel, checkpoint[0] + self.pointLabelOffset)

                pygame.draw.line(screen, self.checkpointColour, checkpoint[0], checkpoint[1], self.checkpointThickness)

        # Finish line
        if len(checkpoints) > 0:
            checkpoint = checkpoints[0]
            pygame.draw.line(screen, self.finishLineColour, checkpoint[0], checkpoint[1], self.finishLineThickness)

    def getSpawnPosition(self):
        if self.curves:
            # Last point in last curve of the track 
            point1 = pygame.Vector2(self.curves[-1][-2].x, self.curves[-1][-2].y)
            point2 = pygame.Vector2(self.curves[-1][-1].x, self.curves[-1][-1].y)

            difference = point2 - point1
            angle = math.degrees(math.atan2(difference.y, difference.x))

            return point1, angle

    def initialiseTrack(self):
        self.curves = self.getCurves()
        self.checkpoints = self.getCheckpoints(self.curves)

        # Adding more curve points when creating final surface
        self.pointFrequency = self.finalPointFrequency
        finalCurves = self.getCurves()

        self.trackSurface = pygame.Surface((TRACK_WIDTH, TRACK_HEIGHT), pygame.SRCALPHA)
        self.drawCircles(self.trackSurface, finalCurves)

        # Finish line
        checkpoint = self.checkpoints[0]
        pygame.draw.line(self.trackSurface, self.finishLineColour, checkpoint[0], checkpoint[1], self.finishLineThickness)

        self.mask = pygame.mask.from_surface(self.trackSurface)
        self.mask.invert()

    def getOverlap(self, x, y, mask):
        overlap = self.mask.overlap(mask, (x, y))
        return overlap

    def checkCollideAtPoint(self, position):
        return self.mask.get_at(position)

    def getFilePath(self):
        return self.filePath or ""

    def exportTrack(self, filePath):
        # Converting from Pygame.Vector2(x, y) -> tuple (x, y) to be stored 
        points = []
        for point in self.points:
            points.append((point.x, point.y))

        output = {
            "Points": points,
            "TrackWidth": self.trackWidth,
            "TrackColour": self.trackColour
        }
        
        with open(f"{TRACKS_PATH}/{filePath}.json", "w") as file:
            json.dump(output, file)

    def importTrack(self, filePath):
        with open(f"{TRACKS_PATH}/{filePath}.json", "r") as file:
            data = json.load(file)
            
            # Converting from tuple (x, y) -> Pygame.Vector2(x, y) to be imported 
            points = data["Points"]

            self.points = []
            for point in points:
                self.points.append(pygame.Vector2(point[0], point[1]))

            self.trackWidth = data["TrackWidth"]
            self.trackColour = data["TrackColour"]
    
    def draw(self, screen, offset):
        screen.blit(self.trackSurface, (-offset, (TRACK_WIDTH, TRACK_HEIGHT)))