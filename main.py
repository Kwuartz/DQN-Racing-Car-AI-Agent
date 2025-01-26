import pygame
from config import FPS, SCREEN_WIDTH, SCREEN_HEIGHT
from classes import Car

pygame.init()

blueCar = pygame.image.load("Assets/BlueCar.png")

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Racing Game")

clock = pygame.time.Clock()

playerCar = Car(300, 300, 0, blueCar)

deltaTime = 1 / FPS

running = True

while running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False
  
  acceleration = 0
  turnDirection = 0
  keys = pygame.key.get_pressed()
  if keys[pygame.K_w]:
    acceleration += 1
  if keys[pygame.K_s]:
    print("hi")
    acceleration += -1
  
  if keys[pygame.K_d]:
    turnDirection += 1
  if keys[pygame.K_a]:
    turnDirection += -1

  playerCar.update(deltaTime, acceleration, turnDirection)

  screen.fill((0, 0, 0))
  playerCar.draw(screen)

  # Update the display
  pygame.display.flip()

  deltaTime = clock.tick(FPS) / 1000

pygame.quit()
