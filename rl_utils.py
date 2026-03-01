import torch 
import random

def huber(x, k=1.0):
    '''
    误差小（|diff| < 1）：用 0.5 * diff²（像 L2，平滑）
    误差大（|diff| ≥ 1）：用 |diff| - 0.5（像 L1，不会因为大误差爆炸）

    普通 L2:  误差越大，惩罚平方级增长（不稳定）
    Huber:    小误差 L2，大误差 L1（更稳健）
    '''
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