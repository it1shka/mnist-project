import pygame
import torch
from models import DenseNetwork

pygame.init()
cell_size = 20
screen = pygame.display.set_mode((28 * cell_size, 28 * cell_size))
pygame.display.set_caption('MNIST test')

drawing = False
image_tensor = torch.zeros((28, 28))

def dist(x1, y1, x2, y2):
  return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

model = torch.load('./models/model_9797.pth')
font = pygame.font.SysFont('Arial', 34)

def update():
  if not drawing: return
  x, y = pygame.mouse.get_pos()

  for i in range(28):
    for j in range(28):
      possible_value = max(0, 1 - 0.5 * dist(x, y, (j + 0.5) * cell_size, (i + 0.5) * cell_size) / cell_size)
      possible_value = min(1, possible_value)
      image_tensor[i][j] = max(image_tensor[i][j].item(), possible_value)

  # display the tensor
  for i in range(28):
    for j in range(28):
      color = image_tensor[i][j].item() * 255
      pygame.draw.rect(screen, (color, color, color), pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size))
  # draw prediction
  # feed should be transposed
  feed = image_tensor.unsqueeze(0).to('mps')
  prediction = model(feed)
  prediction = torch.argmax(prediction, dim=1)
  prediction_text = font.render(str(prediction.item()), True, (255, 255, 255))
  screen.blit(prediction_text, (0, 28 * cell_size - 34))
  pygame.display.update()

while True:
  for event in pygame.event.get():
    match event.type:
      case pygame.QUIT:
        pygame.quit()
        exit(0)
      case pygame.KEYDOWN if event.key == pygame.K_c:
        image_tensor = torch.zeros((28, 28))
        screen.fill((0, 0, 0))
        pygame.display.update()
      case pygame.MOUSEBUTTONDOWN:
        drawing = True
      case pygame.MOUSEBUTTONUP:
        drawing = False
  update()