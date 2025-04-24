import pygame
import math
import os

pygame.init()

from config import FPS, SCREEN_WIDTH, SCREEN_HEIGHT, TRACK_WIDTH, TRACK_HEIGHT, COLOUR_SCHEME, BUTTON_BORDER_THICKNESS, BUTTON_HOVER_THICKNESS, TRACKS_PATH, TOTAL_LAPS, BACKGROUND_COLOUR, BLUE_CAR_IMAGE, RED_CAR_IMAGE, FONT_16, FONT_32, TRAINING_EPISODES, ASPECT_RATIO, MAX_TIMESTEPS, TRAINING_TIMESTEP, SPEED_REWARD
from gui import Container, TextLabel, Button, TextInputBox, Minimap
from cars import Car, CarAgent
from modelNumpy import DQNTrainer
from track import Track


class Game:
    def __init__(self):
        #Initialising screen
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Racing Game")

        self.trainer = DQNTrainer(self)

        self.running = True
        self.clock = pygame.time.Clock()
        self.deltaTime = 1 / FPS

        self.displayMainMenu()

    def resetDeltaTime(self):
        self.deltaTime = 1 / FPS
        self.clock.tick(FPS)

    def displayMainMenu(self):
        #Initialising menu buttons
        playButton = Button(0.1, 0.2, 0.2, 0.1, "Play", FONT_16, COLOUR_SCHEME[0], COLOUR_SCHEME[1],COLOUR_SCHEME[0], BUTTON_BORDER_THICKNESS, BUTTON_HOVER_THICKNESS)
        trackButton = Button(0.1, 0.35, 0.2, 0.1, "Create Track", FONT_16, COLOUR_SCHEME[0], COLOUR_SCHEME[1], COLOUR_SCHEME[0], BUTTON_BORDER_THICKNESS, BUTTON_HOVER_THICKNESS)
        trainButton = Button(0.1, 0.5, 0.2, 0.1, "Train an Agent", FONT_16, COLOUR_SCHEME[0], COLOUR_SCHEME[1], COLOUR_SCHEME[0], BUTTON_BORDER_THICKNESS, BUTTON_HOVER_THICKNESS)
        exitButton = Button(0.1, 0.65, 0.2, 0.1, "Exit", FONT_16, COLOUR_SCHEME[0], COLOUR_SCHEME[1], COLOUR_SCHEME[0], BUTTON_BORDER_THICKNESS, BUTTON_HOVER_THICKNESS)

        buttons = [playButton, trackButton, trainButton, exitButton]

        while self.running:
            # Checking if buttons are hovered
            hoveredButton = None
            for button in buttons:
                if button.updateHovered(pygame.mouse.get_pos()):
                    hoveredButton = button

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Button handling
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
                            trackSelected = self.trackSelection()
                            if trackSelected:
                                self.track.initialiseTrack()
                                self.trainer.train()
                        elif hoveredButton == exitButton:
                            self.running = False

            # Drawing main menu            
            self.screen.fill((8, 132, 28))

            for button in buttons:
                button.draw(self.screen)

            pygame.display.flip()

            # Getting time between frames
            self.deltaTime = self.clock.tick(FPS) / 1000
    
    def trackSelection(self, allowNewTrack=False):
        backButton = Button(0.02, 0.88, 0.1, 0.1, "Back", FONT_16, COLOUR_SCHEME[0], COLOUR_SCHEME[1], COLOUR_SCHEME[0], BUTTON_BORDER_THICKNESS, BUTTON_HOVER_THICKNESS)
        selectButton = Button(0.88, 0.88, 0.1, 0.1, "Select", FONT_16, COLOUR_SCHEME[0], COLOUR_SCHEME[1], COLOUR_SCHEME[0], BUTTON_BORDER_THICKNESS, BUTTON_HOVER_THICKNESS)

        buttons = [backButton, selectButton]

        # Creating the track selection scroll menu
        containerPosition = pygame.Vector2(0.25, 0.05)
        offScreenPosition = pygame.Vector2(2, 2)
        buttonSize = pygame.Vector2(0.48, 0.13)
        buttonPadding = pygame.Vector2(0.01, 0.01 * ASPECT_RATIO)
        tracksPerPage = 6
        scrollIndex = 0

        trackButtonsContainer = Container(containerPosition.x, containerPosition.y, buttonSize.x + buttonPadding.x * 2, (buttonSize.y * (tracksPerPage)) + (buttonPadding.y * (tracksPerPage + 1)), COLOUR_SCHEME[3], COLOUR_SCHEME[0], BUTTON_BORDER_THICKNESS)
        
        trackButtons = []

        # New track button
        if allowNewTrack:
            newTrackButton = Button(0, 0, buttonSize.x, buttonSize.y, "New Track", FONT_16, COLOUR_SCHEME[0], COLOUR_SCHEME[1], COLOUR_SCHEME[0], BUTTON_BORDER_THICKNESS, BUTTON_HOVER_THICKNESS, COLOUR_SCHEME[2])
            trackButtons.append(newTrackButton)

        # Creating track buttons from diirectory
        trackPaths = os.listdir(TRACKS_PATH)
        for trackPath in trackPaths:
            strippedPath = trackPath[:-5]
            trackButton = Button(0, 0, buttonSize.x, buttonSize.y, strippedPath, FONT_16, COLOUR_SCHEME[0], COLOUR_SCHEME[1], COLOUR_SCHEME[0], BUTTON_BORDER_THICKNESS, BUTTON_HOVER_THICKNESS, COLOUR_SCHEME[2])
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

                # Button handling
                elif event.type == pygame.MOUSEBUTTONDOWN:
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

                # Handling scrolling up and down
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
            
            # Only updating the positions of buttons if the user scrolled
            if updateButtons:
                for index, button in enumerate(trackButtons):
                    if index in range(scrollIndex, scrollIndex + tracksPerPage):
                        relativeIndex = index - scrollIndex
                        paddedPosition = containerPosition + buttonPadding
                        button.moveButton(paddedPosition.x, paddedPosition.y + (buttonPadding.y + buttonSize.y) * relativeIndex)
                    else:
                        button.moveButton(offScreenPosition.x, offScreenPosition.y)

            # Drawing buttons and scroll container
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
        # Getting track name
        trackName = self.track.getFilePath()
        editingTrackName = False

        backButton = Button(0.02, 0.88, 0.1, 0.1, "Back", FONT_16, COLOUR_SCHEME[0], COLOUR_SCHEME[1], COLOUR_SCHEME[0], BUTTON_BORDER_THICKNESS, BUTTON_HOVER_THICKNESS)
        saveButton = Button(0.88, 0.88, 0.1, 0.1, "Save", FONT_16, COLOUR_SCHEME[0], COLOUR_SCHEME[1], COLOUR_SCHEME[0], BUTTON_BORDER_THICKNESS, BUTTON_HOVER_THICKNESS)
        trackNameLabel = TextLabel(0.88, 0.71, 0.1, 0.05, "TRACK NAME:", FONT_32, COLOUR_SCHEME[0])
        trackNameBox = TextInputBox(0.88, 0.76, 0.1, 0.1, trackName, FONT_16, COLOUR_SCHEME[0], COLOUR_SCHEME[1], COLOUR_SCHEME[0], BUTTON_BORDER_THICKNESS, BUTTON_HOVER_THICKNESS, COLOUR_SCHEME[2])

        elements = [backButton, saveButton, trackNameBox]

        # Scaling down the track to fit on the screen
        zoom = int(TRACK_WIDTH / SCREEN_WIDTH)
        trackSurface = pygame.Surface((TRACK_WIDTH, TRACK_HEIGHT))
        selectedPoint = None

        firstDraw = True
        editorRunning = True
        while self.running and editorRunning:
            updateTrack = False
            hoveredElement = None
            for element in elements:
                if element.updateHovered(pygame.mouse.get_pos()):
                    hoveredElement = element

            # Scaling the mouse position so nodes are placed down accurately
            mousePosition = pygame.mouse.get_pos()
            mousePosition = pygame.Vector2(mousePosition[0], mousePosition[1])
            scaledMousePosition = mousePosition * zoom

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    editingTrackName = False
                    trackNameBox.setSelected(editingTrackName)

                    if event.button == 1:
                        updateTrack = True
                        if hoveredElement:
                            # Button handling
                            if hoveredElement == backButton:
                                editorRunning = False
                            elif hoveredElement == saveButton:
                                if selectedPoint is None:
                                    trackName = trackNameBox.getText()
                                    if trackName == "":
                                        trackNameLabel.textColour = COLOUR_SCHEME[5]
                                        trackNameLabel.updateText("INVALID TRACK NAME:")
                                    elif not self.track.curves:
                                        trackNameLabel.textColour = COLOUR_SCHEME[5]
                                        trackNameLabel.updateText("INVALID TRACK:")
                                    else:
                                        self.track.initialiseTrack()
                                        self.track.exportTrack(trackName)
                                        editorRunning = False

                            # User is editing the track name
                            elif hoveredElement == trackNameBox:
                                editingTrackName = True
                                trackNameBox.setSelected(editingTrackName)

                        elif selectedPoint is not None:
                            # Placing down a point that was being moved
                            selectedPoint = None
                        else:
                            # Point creation/selection (Left click)
                            if (hoveredPoint := self.track.getHoveredPoint(scaledMousePosition)) is not None:
                                # Selecting an existing point
                                selectedPoint = hoveredPoint
                            else:
                                # Adding a new point
                                self.track.addPoint(scaledMousePosition)

                    elif event.button == 3:
                        # Point deletion (Right click)
                        updateTrack = True
                        if (hoveredPoint := self.track.getHoveredPoint(scaledMousePosition)) is not None:
                            # Removing the hovered point
                            self.track.removePoint(hoveredPoint)
                        else:
                            # Removing the last point that was placed down
                            self.track.removePoint()

                elif event.type == pygame.KEYDOWN and editingTrackName:
                    # Updating the track name
                    trackNameBox.update(event)

            if selectedPoint is not None:
                # Moving the selected point with the mouse cursor
                updateTrack = True
                self.track.movePoint(selectedPoint, scaledMousePosition)
            
            # Making sure the track is always drawn on the first frame
            if firstDraw:
                updateTrack = True
                firstDraw = False

            # Only update the track if neccessary to save computing power
            if updateTrack:
                trackSurface.fill(BACKGROUND_COLOUR)
                self.track.drawEditor(trackSurface)

            # Scaling the track surface and drawing it
            scaledTrackSurface = pygame.transform.scale(trackSurface, (SCREEN_WIDTH, SCREEN_HEIGHT))
            self.screen.blit(scaledTrackSurface, (0, 0))

            for element in elements:
                element.draw(self.screen)

            trackNameLabel.draw(self.screen)

            pygame.display.flip()

            self.deltaTime = self.clock.tick(FPS) / 1000

    def gameLoop(self):
        minimapSize = pygame.Vector2(0.2, 0.2)
        minimap = Minimap(0, 0, minimapSize.x, minimapSize.y, self.track, COLOUR_SCHEME[1], BUTTON_BORDER_THICKNESS)
        
        containerSize = pygame.Vector2(0.05, 0.05)
        containerPosition = minimapSize - containerSize
        gameInfoContainer = Container(containerPosition.x, containerPosition.y, containerSize.x, containerSize.y, COLOUR_SCHEME[2], COLOUR_SCHEME[1], BUTTON_BORDER_THICKNESS)
        lapLabel = TextLabel(containerPosition.x, containerPosition.x, containerSize.x, containerSize.y, f"Lap 1/{TOTAL_LAPS}", FONT_16, COLOUR_SCHEME[0])

        spawnPoint, spawnAngle = self.track.getSpawnPosition()

        playerCar = Car(spawnPoint.x, spawnPoint.y, spawnAngle, BLUE_CAR_IMAGE)
        agentCar = CarAgent(spawnPoint.x, spawnPoint.y, spawnAngle, RED_CAR_IMAGE, False)

        lap = 1
        cameraOffset = pygame.Vector2(0, 0)

        self.resetDeltaTime()

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

            playerCar.update(self.deltaTime, acceleration, turnDirection, self.track)
            agentCar.update(self.deltaTime, self.track)

            # Check if game over
            newLap = max(playerCar.lap, agentCar.lap)
            if lap != newLap:
                lap = newLap
                lapLabel.updateText(f"Lap {lap}/{TOTAL_LAPS}")
            if lap > TOTAL_LAPS:
                gameRunning = False

            cameraOffset = playerCar.getCameraOffset(cameraOffset)
            
            self.screen.fill(BACKGROUND_COLOUR)
            self.track.draw(self.screen, cameraOffset)

            playerCar.draw(self.screen, cameraOffset)
            agentCar.draw(self.screen, cameraOffset)

            minimap.draw(self.screen, playerCar, agentCar)

            gameInfoContainer.draw(self.screen)
            lapLabel.draw(self.screen)

            pygame.display.flip()

            self.deltaTime = self.clock.tick(FPS) / 1000

    def trainingMenu(self):
        percentage = self.trainer.episode / TRAINING_EPISODES
        barPadding = pygame.Vector2(0.01, 0.01 * ASPECT_RATIO)
        barSize = pygame.Vector2(0.5, 0.1)
        barPosition = pygame.Vector2(0.25, 0.3)

        outerBar = Container(barPosition.x, barPosition.y, barSize.x, barSize.y, COLOUR_SCHEME[1], COLOUR_SCHEME[0], BUTTON_BORDER_THICKNESS)
        innerBar = Container(barPosition.x + barPadding.x, barPosition.y + barPadding.y, (barSize.x - barPadding.x * 2) * percentage, barSize.y - barPadding.y * 2, COLOUR_SCHEME[0])

        labelPosition = barPosition + pygame.Vector2(0, barSize.y)
        episodeLabel = TextLabel(labelPosition.x, labelPosition.y, barSize.x, barSize.y, f"{self.trainer.episode} / {TRAINING_EPISODES}", FONT_32, COLOUR_SCHEME[0])

        self.screen.fill(BACKGROUND_COLOUR)
            
        outerBar.draw(self.screen)
        innerBar.draw(self.screen)

        episodeLabel.draw(self.screen)

        pygame.display.flip()

        pygame.display.flip()

    def visualizeEpisode(self):
        skipButton = Button(0.88, 0.88, 0.1, 0.1, "Skip", FONT_16, COLOUR_SCHEME[0], COLOUR_SCHEME[1], COLOUR_SCHEME[0], BUTTON_BORDER_THICKNESS, BUTTON_HOVER_THICKNESS)

        elements = [skipButton]
        
        spawnPoint, spawnAngle = self.track.getSpawnPosition()
        agentCar = CarAgent(spawnPoint.x, spawnPoint.y, spawnAngle, RED_CAR_IMAGE, self.trainer.policyNet, False)

        self.resetDeltaTime()
        timeSteps = 0
        maxTimeSteps = MAX_TIMESTEPS / TRAINING_TIMESTEP

        visualisationRunning = True
        while visualisationRunning and self.running:
            hoveredElement = None
            for element in elements:
                if element.updateHovered(pygame.mouse.get_pos()):
                    hoveredElement = element

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if hoveredElement:
                            if hoveredElement == skipButton:
                                visualisationRunning = False
            
            # Select best action
            crashed = agentCar.update(self.deltaTime, self.track)
            timeSteps += 1
            
            if timeSteps % 25 == 0:
                print("SPEED RWD:", SPEED_REWARD * agentCar.speed)
                print("SPEED:", agentCar.speed)
                print("FPS", 1 / self.deltaTime)

            if crashed or agentCar.lap > 1 or timeSteps >= maxTimeSteps:
                visualisationRunning = False

            trackSurface = pygame.Surface((TRACK_WIDTH, TRACK_HEIGHT), pygame.SRCALPHA)
            trackSurface.fill(BACKGROUND_COLOUR)

            self.track.draw(trackSurface, pygame.Vector2(0, 0))
            agentCar.draw(trackSurface, pygame.Vector2(0, 0))

            scaledTrackSurface = pygame.transform.scale(trackSurface, (SCREEN_WIDTH, SCREEN_HEIGHT))
            self.screen.blit(scaledTrackSurface, (0, 0))

            for element in elements:
                element.draw(self.screen)

            pygame.display.flip()

            self.deltaTime = self.clock.tick(FPS) / 1000

if __name__ == "__main__":
    Game()