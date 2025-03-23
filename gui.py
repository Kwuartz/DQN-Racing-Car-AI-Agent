from config import SCREEN_WIDTH, SCREEN_HEIGHT, TRACK_WIDTH, TRACK_HEIGHT, BACKGROUND_COLOUR

import pygame

class GuiElement:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(SCREEN_WIDTH * x, SCREEN_HEIGHT * y, SCREEN_WIDTH * width, SCREEN_HEIGHT * height)

class TextLabel(GuiElement):
    def __init__(self, x, y, width, height, text, font, textColour):
        super().__init__(x, y, width, height)
        
        self.font = font
        self.textColour = textColour
        self.text = font.render(text, True, textColour)

    def draw(self, screen):
        screen.blit(self.text, self.text.get_rect(center=self.rect.center))

    def updateText(self, newText):
        self.text = self.font.render(newText, True, self.textColour)

class Minimap(GuiElement):
    def __init__(self, x, y, width, height, track, borderColour, borderThickness):
        super().__init__(x, y, width, height)
        self.surface = pygame.Surface((self.rect.width, self.rect.height))
        self.trackSurface = pygame.Surface((TRACK_WIDTH, TRACK_HEIGHT))

        self.trackSurface.fill(BACKGROUND_COLOUR)
        track.draw(self.trackSurface, pygame.Vector2(0, 0))

        self.scale = min(self.rect.width / TRACK_WIDTH, self.rect.height / TRACK_HEIGHT)
        self.trackSurface = pygame.transform.scale(self.trackSurface, (self.rect.width, self.rect.height))

        self.borderColour = borderColour
        self.borderThickness = borderThickness

        self.playerColour = (255, 0, 0)
        self.agentColour = (255, 255, 0)
        self.dotSize = 3

    def draw(self, screen, player, agent):
        self.surface.blit(self.trackSurface, (0, 0))
        
        pygame.draw.rect(self.surface, self.borderColour, self.rect, self.borderThickness)
        pygame.draw.circle(self.surface, self.playerColour, (player.x * self.scale, player.y * self.scale), self.dotSize)
        pygame.draw.circle(self.surface, self.agentColour, (agent.x * self.scale, agent.y * self.scale), self.dotSize)

        screen.blit(self.surface, self.rect)

class Container(GuiElement):
    def __init__(self, x, y, width, height, backgroundColour, borderColour=(0,0,0), borderThickness=0):
        super().__init__(x, y, width, height)
        self.backgroundColour = backgroundColour        
        self.borderColour = borderColour
        self.borderThickness = borderThickness

    def draw(self, screen):
        pygame.draw.rect(screen, self.backgroundColour, self.rect)

        if self.borderThickness > 0:
            pygame.draw.rect(screen, self.borderColour, self.rect, self.borderThickness)

class Button(TextLabel):
    def __init__(self, x, y, width, height, text, font, textColour, backgroundColour, borderColour=(0,0,0), borderThickness=0, hoverBorderThickness=0, selectedBackgroundColour=(0,0,0)):
        super().__init__(x, y, width, height, text, font, textColour)
        
        self.font = font
        self.textColour = textColour
        self.text = font.render(text, True, textColour)

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
        super().draw(screen)

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
    def __init__(self, x, y, width, height, text, font, textColour, backgroundColour, borderColour=(0,0,0), borderThickness=0, hoverBorderThickness=0, selectedBackgroundColour=(0,0,0)):
        super().__init__(x, y, width, height, text, font, textColour, backgroundColour, borderColour, borderThickness, hoverBorderThickness, selectedBackgroundColour)
        self.textContent = text

    def update(self, event):
        if event.key == pygame.K_BACKSPACE:
            self.textContent = self.textContent[:-1]
        elif event.unicode.isalnum():
            self.textContent += event.unicode

        self.updateText(self.textContent)

    def getText(self):
        return self.textContent