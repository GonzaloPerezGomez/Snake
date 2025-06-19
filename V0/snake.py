from enum import Enum
import pygame
import numpy as np
import random
import copy
import math

pygame.init()

SPEED = 100000000000000000000000000000000000000000000000
BLOCK_SIZE = 20
BLACK = [0,0,0]
GREEN = [0,255,0]
DARK_GREEN = [0,80,0]
RED = [255,0,0]
WHITE = [255, 255, 255]
PADDING = 6
FOOD_REWARD = 10
DIE_REWARD = -10

class Direc(Enum):
    NORTH = 0
    WEST = 1
    SOUTH = 2
    EAST = 3
    
class Point():
    
    def __init__(self, x, y, dir=None):
        self.x = x
        self.y = y
        self.dir = dir
        
    def __eq__(self, p):
        return self.x == p.x and self.y == p.y


class Snake():
    
    def __init__(self, w=640, h=480):
        
        self.font = pygame.font.Font('arial.ttf', 25)
        
        self.w = w
        self.h = h
        self.action_space = [
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
        ]
        
        self.screen = pygame.display.set_mode((self.w , self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        
    def reset(self):
        # Set the initial move history
        self.move_history = [Direc.NORTH, Direc.NORTH, Direc.NORTH]
        # Reset the score counter
        self.score = 0
        # Set the head of the snake in the center of the frame
        self.head = Point((((self.w - BLOCK_SIZE) // BLOCK_SIZE) - 1) / 2, (((self.h - BLOCK_SIZE) // BLOCK_SIZE) - 1) / 2, self.move_history[0])
        # Create the snake with a head and 2 blocks of body
        self.snake = [self.head,
                      Point(self.head.x, self.head.y + 1, self.move_history[1]),
                      Point(self.head.x, self.head.y + 2, self.move_history[2])]
        # Create the food
        self.food_spawn()
        
        self.dist = math.hypot(self.head.x - self.food.x, self.head.y - self.food.y)
        
        # Refresh the frame
        frame_array = self.update()
        return frame_array
        
    def step(self, action: list):
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
                
            keystate = pygame.key.get_pressed()
            if keystate[pygame.K_UP]:
                action = [1,0,0,0]
            elif keystate[pygame.K_LEFT]:
                action = [0,1,0,0]
            elif keystate[pygame.K_DOWN]:
                action = [0,0,1,0]
            elif keystate[pygame.K_RIGHT]:
                action = [0,0,0,1]
                
        # Update the move_history
        if np.array_equal(action, [1,0,0,0]):
            new_dir = Direc.NORTH
        elif np.array_equal(action, [0,1,0,0]):
            new_dir = Direc.WEST
        elif np.array_equal(action, [0,0,1,0]):
            new_dir = Direc.SOUTH
        elif np.array_equal(action, [0,0,0,1]):
            new_dir = Direc.EAST
        else:
            new_dir = self.move_history[0]
            
        self.move_history.insert(0, new_dir)
                    
        #Update the directions
        for i, p in enumerate(self.snake):
            p.dir = self.move_history[i]
                
        # Move
        future_snake = copy.deepcopy(self.snake)
        
        for p in future_snake:
            if p.dir == Direc.NORTH:
                p.y -= 1
            elif p.dir == Direc.WEST:
                p.x -= 1
            elif p.dir == Direc.SOUTH:
                p.y += 1
            elif p.dir == Direc.EAST:
                p.x += 1
                
        # Check collisions    
        game_over, reward = self.is_collision(future_snake)
        
        # Check for food eaten
        if self.food == future_snake[0]:
            self.score += 1
            reward = FOOD_REWARD
            last = self.snake[-1]
            future_snake.append(Point(last.x, last.y, last.dir))
            self.food_spawn()
            
        # Reward for the proximity to the food
        self.next_dist = math.hypot(self.head.x - self.food.x, self.head.y - self.food.y)
        if self.next_dist > self.dist:
            reward -= 0.2
        else:
            reward += 0.2
            
        self.dist = self.next_dist
        
        # Price for each frame
        reward += 0.05
        
        if not game_over:
            self.snake = future_snake
            # Update frame
            self.clock.tick(SPEED)
            
        frame_array = self.update()
        
        return frame_array, reward, game_over
        
        
    def food_spawn(self):
        # Set the food in a random location
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE)
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE)
        self.food = Point(x, y)
        if self.food in self.snake:
            self.food_spawn()
            
    def is_collision(self, snake: list[Point]):
        p = snake[0]
        b = snake[1:]
        # Check collision with the limits
        if p.y < 0 or p.y > (self.h - BLOCK_SIZE) // BLOCK_SIZE:
            return True, -DIE_REWARD
        elif p.x < 0 or p.x > (self.w - BLOCK_SIZE) // BLOCK_SIZE:
            return True, DIE_REWARD
        # Check collision with the body
        elif p in b:
            return True, DIE_REWARD
        
        return False, 0
        
    def update(self):
        # Reset the frame
        self.screen.fill(BLACK)
        # Draw the snake
        for p in self.snake:
            pygame.draw.rect(self.screen, DARK_GREEN, pygame.Rect(p.x * BLOCK_SIZE, p.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            if p == self.snake[0]:
                pygame.draw.rect(self.screen, DARK_GREEN, pygame.Rect((p.x * BLOCK_SIZE) + PADDING / 2, (p.y * BLOCK_SIZE) + PADDING / 2, BLOCK_SIZE - PADDING, BLOCK_SIZE - PADDING))
            else:
                pygame.draw.rect(self.screen, GREEN, pygame.Rect((p.x * BLOCK_SIZE) + PADDING / 2, (p.y * BLOCK_SIZE) + PADDING / 2, BLOCK_SIZE - PADDING, BLOCK_SIZE - PADDING))
        # Draw the food
        pygame.draw.rect(self.screen, WHITE, pygame.Rect(self.food.x * BLOCK_SIZE, self.food.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        
        text = self.font.render("Score: " + str(self.score), True, WHITE)
        self.screen.blit(text, [0, 0])
        
        pygame.display.flip()
        frame_array = pygame.surfarray.array3d(self.screen)
        return frame_array
    