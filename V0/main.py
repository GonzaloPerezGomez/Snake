import random
import torch
from agent import SnakeV0, SnakeV1, SnakeV2
from snake import Snake
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from utils import init_plot, update_plot
from torchvision import transforms
from PIL import Image
import numpy as np

EPSILON_START = 0.99
EPSILON_END = 0.01
EPSILON_DECAY = 0.9999
GAMMA = 0.99


def main():
    game = Snake()
    model = SnakeV2(in_channels=3,
                  hidden_units=10,
                  output_shape=len(game.action_space[0]))
    
    # Loss function
    loss_fn = nn.MSELoss()

    # Optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr = 1e-4)
    
    train_epochs = 100000
    trainV2(model, game, loss_fn, optimizer, train_epochs)


def trainV0(model: SnakeV0, env: Snake, loss_fn: nn.MSELoss, optimizer: optim.SGD, epochs: int):
    
    fig, axs = init_plot()
    scores = []
    epsilons = []
    losses = []

    for epoch in tqdm(range(epochs)):
        
        state = env.reset()
        game_over = False
        steps = 0
        loss_sum = 0
        
        while not game_over:
             
            state_tensor = torch.from_numpy(state).permute(2,0,1).unsqueeze(0).to(torch.float) / 255.0
            
            # Update the epsilon value
            epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** epoch))
            # Use epsilon-greedy
            random_float = random.random()
            if random_float < epsilon:
                # Explore
                action = random.sample(env.action_space, k=1)
            else:
                # Act greedy (explotation)
                action = model(state_tensor)
            
            action_idx = torch.argmax(torch.tensor(action)).item()    
            aux = torch.zeros(4)
            aux[action_idx] = 1
            action = aux
                
            # We perform the new action
            next_state, reward, game_over = env.step(action)
            
            next_state_tensor = torch.from_numpy(next_state).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                
            # Calculate the current q_value
            q_values = model(state_tensor)
            
            # Calculate the target q_value
            with torch.no_grad():
                next_q_values = model(next_state_tensor)
                max_next_q = next_q_values.max(1)[0].item()
                target_value = reward + GAMMA * max_next_q * (0 if game_over else 1)
                
            target_q_values = q_values.clone().detach()
            target_q_values[0, action_idx] = target_value 
            
            # Calculate the loss
            loss = loss_fn(q_values, target_q_values)
            print(f"Loss: {loss.detach().cpu().item()}")
            loss_sum += loss.detach().cpu().item()
            print(f"Loss Sum: {loss_sum}")
            # Optimier zero grad
            optimizer.zero_grad()
            
            # Back propagation
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            state = next_state
            steps += 1
            
        scores.append(env.score)
        epsilons.append(epsilon)
        losses.append(loss_sum / steps)
        
        update_plot(fig, axs, scores, epsilons, losses)
        
    fig.savefig('grafico_prueba.png')
    torch.save(model.state_dict(), 'modelo_de_prueba.pth')
    
def trainV1(model: SnakeV0, env: Snake, loss_fn: nn.MSELoss, optimizer: optim.SGD, epochs: int):
    
    fig, axs = init_plot()
    scores = []
    epsilons = []
    losses = []

    for epoch in tqdm(range(epochs)):
        
        state = env.reset()
        game_over = False
        steps = 0
        loss_sum = 0
        
        while not game_over:
             
            state_tensor = torch.from_numpy(state).permute(2,0,1).unsqueeze(0).to(torch.float) / 255.0
            food_pos = torch.tensor([[env.food.x / env.w, env.food.y / env.h]], dtype=torch.float32)
            
            # Update the epsilon value
            epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** epoch))
            # Use epsilon-greedy
            random_float = random.random()
            if random_float < epsilon:
                # Explore
                action = random.sample(env.action_space, k=1)
            else:
                # Act greedy (explotation)
                action = model(state_tensor, food_pos)
            
            action_idx = torch.argmax(torch.tensor(action)).item()    
            aux = torch.zeros(4)
            aux[action_idx] = 1
            action = aux
                
            # We perform the new action
            next_state, reward, game_over = env.step(action)
            
            next_state_tensor = torch.from_numpy(next_state).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                
            # Calculate the current q_value
            q_values = model(state_tensor, food_pos)
            
            # Calculate the target q_value
            with torch.no_grad():
                next_q_values = model(next_state_tensor, food_pos)
                max_next_q = next_q_values.max(1)[0].item()
                target_value = reward + GAMMA * max_next_q * (0 if game_over else 1)
                
            target_q_values = q_values.clone().detach()
            target_q_values[0, action_idx] = target_value 
            
            # Calculate the loss
            loss = loss_fn(q_values, target_q_values)
            print(f"Loss: {loss.detach().cpu().item()}")
            loss_sum += loss.detach().cpu().item()
            print(f"Loss Sum: {loss_sum}")
            # Optimier zero grad
            optimizer.zero_grad()
            
            # Back propagation
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            state = next_state
            steps += 1
            
        scores.append(env.score)
        epsilons.append(epsilon)
        losses.append(loss_sum / steps)
        
        update_plot(fig, axs, scores, epsilons, losses)
        
    fig.savefig('grafico_prueba.png')
    torch.save(model.state_dict(), 'modelo_de_prueba.pth')
    
    
def trainV2(model: SnakeV0, env: Snake, loss_fn: nn.MSELoss, optimizer: optim.SGD, epochs: int):
    
    fig, axs = init_plot()
    scores = []
    epsilons = []
    losses = []
    
    transform = transforms.Compose([
        transforms.Resize((120, 90)),
        transforms.ToTensor()
    ])


    for epoch in tqdm(range(epochs)):
        
        state = env.reset()
        game_over = False
        steps = 0
        loss_sum = 0
        
        while not game_over:
             
            state_tensor = transform(Image.fromarray(state.astype(np.uint8))).unsqueeze(0)
            food_pos = torch.tensor([[env.food.x / env.w, env.food.y / env.h]], dtype=torch.float32)
            head_pos = torch.tensor([[env.head.x / env.w, env.head.y / env.h]], dtype=torch.float32)
            dir = torch.zeros(4)
            dir[env.head.dir.value] = 1
            head_dir = dir.unsqueeze(0)
            
            info_tensor = torch.cat([food_pos, head_pos, head_dir], dim=1).to(torch.float32)
            
            # Update the epsilon value
            epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** epoch))
            # Use epsilon-greedy
            random_float = random.random()
            if random_float < epsilon:
                # Explore
                action = random.sample(env.action_space, k=1)
            else:
                # Act greedy (explotation)
                action = model(state_tensor, info_tensor)
            
            action_idx = torch.argmax(torch.tensor(action)).item()    
            aux = torch.zeros(4)
            aux[action_idx] = 1
            action = aux
                
            # Perform the new action
            next_state, reward, game_over = env.step(action)
            next_state_tensor = transform(Image.fromarray(next_state.astype(np.uint8))).unsqueeze(0)
            
            # Calculate the current q_value
            q_values = model(state_tensor, info_tensor)
            
            # Calculate the target q_value
            with torch.no_grad():
                next_q_values = model(next_state_tensor, info_tensor)
                max_next_q = next_q_values.max(1)[0].item()
                target_value = reward + GAMMA * max_next_q * (0 if game_over else 1)
                
            target_q_values = q_values.clone().detach()
            target_q_values[0, action_idx] = target_value 
            
            # Calculate the loss
            loss = loss_fn(q_values, target_q_values)
            loss_sum += loss.detach().cpu().item()
            
            # Optimier zero grad
            optimizer.zero_grad()
            
            # Back propagation
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            state = next_state
            steps += 1
            
        scores.append(env.score)
        epsilons.append(epsilon)
        losses.append(loss_sum / steps)
        
        update_plot(fig, axs, scores, epsilons, losses)
        
    fig.savefig('grafico_prueba.png')
    torch.save(model.state_dict(), 'modelo_de_prueba.pth')
            

def test():
    return None

if __name__ == "__main__": main()