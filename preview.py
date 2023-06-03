import pygame
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from models import DenseNetwork

transform = ToTensor()
dataset = MNIST(root='data/', train=True, transform=transform)

# create a window in pygame
cell_size = 20
pygame.init()
screen = pygame.display.set_mode((28 * cell_size, 28 * cell_size))
pygame.display.set_caption('MNIST preview')
font = pygame.font.SysFont('Arial', 20)

model = torch.load('./models/model_9797.pth')

# display image
def display_image(index):
  global dataset
  image_tensor, label = dataset[index]
  image_tensor = torch.squeeze(image_tensor)
  image_height, image_width = image_tensor.shape
  for i in range(image_height):
    for j in range(image_width):
      color = image_tensor[i][j].item() * 255
      # pygame.draw.rect(screen, (color, color, color), pygame.Rect(i * cell_size, j * cell_size, cell_size, cell_size))
      pygame.draw.rect(screen, (color, color, color), pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size))
  # draw label
  label_text = font.render(str(label), True, (255, 255, 255))
  screen.blit(label_text, (0, 0))  
  # draw prediction
  feed = image_tensor.unsqueeze(0).to('mps')
  prediction = model(feed)
  prediction = torch.argmax(prediction, dim=1)
  prediction_text = font.render(str(prediction.item()), True, (255, 255, 255))
  screen.blit(prediction_text, (0, 28 * cell_size - 20))
  pygame.display.update()

index = 0
display_image(index)

while True:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      pygame.quit()
      exit(0)
    if event.type == pygame.KEYDOWN:
      if event.key == pygame.K_RIGHT:
        index += 1
        index %= len(dataset)
        display_image(index)
      if event.key == pygame.K_LEFT:
        index -= 1
        index = max(index, 0)
        display_image(index)
