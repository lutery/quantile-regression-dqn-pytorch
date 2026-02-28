import torch 
import random

def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, state, action, next_state, reward, done):
        transition = (
            torch.as_tensor(state, dtype=torch.float32).unsqueeze(0),
            torch.tensor([action], dtype=torch.long),
            torch.as_tensor(next_state, dtype=torch.float32).unsqueeze(0),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor([done], dtype=torch.float32),
        )
        self.memory.append(transition)
        if len(self.memory) > self.capacity: del self.memory[0]

    def sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)
        batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*sample)
        
        batch_state = torch.cat(batch_state, dim=0)
        batch_action = torch.cat(batch_action, dim=0)
        batch_reward = torch.cat(batch_reward, dim=0)
        batch_done = torch.cat(batch_done, dim=0)
        batch_next_state = torch.cat(batch_next_state, dim=0)
        
        return batch_state, batch_action, batch_reward.unsqueeze(1), batch_next_state, batch_done.unsqueeze(1)
    
    def __len__(self):
        return len(self.memory)