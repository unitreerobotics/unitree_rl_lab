'''
Generalized training class. Each specific domain will need to override this class.
'''
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
from .models.neural_lyapunov_model import NeuralLyapunovModel

from tensordict import TensorDict

class TrainingBuffer:
    def __init__(self, dataloader: DataLoader, ce_buffer, ratio_original: float = 0.7):
        self.dataloader = dataloader
        self.ce_buffer = ce_buffer
        self.ratio_original = ratio_original

    def __iter__(self):
        for obs_td, actions, next_td, sim_states in self.dataloader:
            # If no CE buffer or empty, yield original batch
            if self.ce_buffer is None or getattr(self.ce_buffer, 'size', 0) == 0:
                yield obs_td, actions, next_td, sim_states
                continue

            batch_size = actions.shape[0]
            num_ce = int((1.0 - self.ratio_original) * batch_size)

            ce_obs_td, ce_actions, ce_next_td, ce_sim_states = self.ce_buffer.sample(num_ce)

            dev = actions.device
            ce_obs_td = ce_obs_td.to(dev)
            ce_next_td = ce_next_td.to(dev)
            ce_actions = ce_actions.to(dev)

            rand_idx = torch.randperm(batch_size, device=dev)[:num_ce]
            obs_td[rand_idx] = ce_obs_td
            next_td[rand_idx] = ce_next_td
            actions[rand_idx] = ce_actions
            sim_states = list(sim_states)
            for j, i in enumerate(rand_idx.tolist()):
                sim_states[i] = ce_sim_states[j]

            yield obs_td, actions, next_td, sim_states

class Trainer():
    def __init__(self, policy, lr, loss_fn, dt, n_inputs, hidden_sizes=None, circle_tuning_loss_fn=None, falsifier=None, device=None, lyapunov_by_construction=True, alpha=0.0001):

        self.policy = policy
        self.lr = lr
        self.lyapunov_loss = loss_fn
        self.circle_tuning_loss = circle_tuning_loss_fn
        self.falsifier = falsifier
        self.dt = dt
        self.device = device if device is not None else 'cpu'
        
        self.n_inputs = n_inputs
        self.hidden_sizes = hidden_sizes if hidden_sizes is not None else [64, 64]
        
        self.model = NeuralLyapunovModel(n_inputs, self.hidden_sizes)
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    
        self.lyapunov_by_construction = lyapunov_by_construction
        if self.lyapunov_by_construction:
            print("Using Lyapunov By Construction with circle tuning alpha={}.".format(alpha))
        self.alpha = alpha

    def get_approx_lie_derivative(self, V_candidate, V_candidate_next):
        '''
        Calculates L_V = ∑∂V/∂xᵢ*fᵢ by forward finite difference
                    L_V = (V' - V) / dt
        '''
        return (V_candidate_next - V_candidate) / self.dt

    def approx_f_value(self, X, X_prime):
        # Approximate f value with S, a, S'
        y = (X_prime - X) / self.dt
        return y

    def adjust_learning_rate(self, decay_rate=.9):
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] * decay_rate
    
    def reset_learning_rate(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr
    
    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, load_path):
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        return self.model
    
    def get_lyapunov_output(self, X, x_0):
        phi_x = self.model(X)
        phi_x0 = self.model(x_0)
        if self.lyapunov_by_construction:
            term_1 = torch.pow(torch.norm(phi_x - phi_x0, dim=1), 2)
            term_2 = self.alpha * torch.pow(torch.norm(X - x_0, dim=1), 2)
            V = term_1 + term_2
        else:
            V = phi_x

        return V

    def evaluate_test_set(self, test_dataloader, x_0):
        self.model.eval()
        total_test_loss = 0.0
        total_test_V_loss = 0.0
        total_test_lie_loss = 0.0
        total_test_eq_loss = 0.0
        total_test_circle_loss = 0.0
        num_test_batches = 0
        
        with torch.no_grad():
            for batch in test_dataloader:
                X, u, X_prime, sim_states = batch
                X = X.to(self.device)
                X_prime = X_prime.to(self.device)
                
                # get lyapunov function from model
                V_candidate = self.get_lyapunov_output(self.process_state(X), x_0)
                if self.lyapunov_by_construction:
                    # already baked into lyapunov function by construction
                    V_X0 = torch.tensor(0.0)
                else:
                    V_X0 = self.model(x_0)
                
                V_candidate_prime = self.get_lyapunov_output(self.process_state(X_prime), x_0)
                
                L_V = self.get_approx_lie_derivative(V_candidate, V_candidate_prime).squeeze()
                loss = self.lyapunov_loss(V_candidate, L_V, V_X0)
                circle_loss = 0.0
                if self.circle_tuning_loss is not None:
                    circle_loss = self.circle_tuning_loss(X, V_candidate)
                    loss += circle_loss
                
                total_test_loss += loss.item()
                total_test_V_loss += self.lyapunov_loss.V_loss
                total_test_lie_loss += self.lyapunov_loss.lie_loss
                total_test_eq_loss += self.lyapunov_loss.eq_loss
                total_test_circle_loss += circle_loss
                num_test_batches += 1
        
        avg_test_loss = total_test_loss / num_test_batches
        avg_test_V_loss = total_test_V_loss / num_test_batches
        avg_test_lie_loss = total_test_lie_loss / num_test_batches
        avg_test_eq_loss = total_test_eq_loss / num_test_batches
        avg_test_circle_loss = total_test_circle_loss / num_test_batches
        
        return avg_test_loss, avg_test_V_loss, avg_test_lie_loss, avg_test_eq_loss, avg_test_circle_loss
           
    def process_state(self, X):
        '''
        Custom function for input to lyapunov model.
        X: (batch, num_states)
        '''
        return X

    def train(self, dataset, x_0, epochs=1000, verbose=False, step_size=100, decay_rate=1., batch_size=64, test_split=0.2, run_name='lyapunov', pct_counterexamples=0.3):
        
        # Split dataset into train and test
        dataset_size = len(dataset)
        test_size = int(test_split * dataset_size)
        train_size = dataset_size - test_size
        
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        # initialize dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_tensordict  
        )

        training_buffer = TrainingBuffer(
            train_dataloader,
            self.falsifier.buffer if self.falsifier is not None else None,
            ratio_original=1.0-pct_counterexamples,
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_tensordict
        )
    
        print(f"Train Data Size: {len(train_dataset)}")
        print(f"Test Data Size: {len(test_dataset)}")
        if self.falsifier is not None:
            print(f"Percentage of Counterexamples: {pct_counterexamples}")
        no_counterexamples_ct = 0
        loss_list = []
        x_0 = x_0.to(self.device)

        # Initialize wandb
        config = {
            "epochs": epochs,
            "learning_rate": self.lr,
            "step_size": step_size,
            "decay_rate": decay_rate,
            "batch_size": batch_size,
            "test_split": test_split
        }
        
        if decay_rate < 1:
            run_name += f"_ss_{step_size}_dr_{decay_rate}"

        run = wandb.init(
            project="lyapunov-g1",
            name=run_name,
            config=config,
        )

        pbar = tqdm(range(1, epochs+1), desc="Training")
        for epoch in pbar:
            self.model.train()

            if (epoch + 1) % step_size == 0:
                self.adjust_learning_rate(decay_rate)
            
            total_loss = 0.0
            total_V_loss = 0.0
            total_lie_loss = 0.0
            total_eq_loss = 0.0
            total_circle_loss = 0.0
            num_batches = 0
            # number of ces per epoch
            num_counterexamples = 0
            for batch in training_buffer:
                X, _, X_prime, sim_states = batch
                X = X.to(self.device)
                X_prime = X_prime.to(self.device)
                self.optimizer.zero_grad()

                # get lyapunov function from model
                V_candidate = self.get_lyapunov_output(self.process_state(X), x_0)
                if self.lyapunov_by_construction:
                    # already baked into lyapunov function by construction
                    V_X0 = torch.tensor(0.0)
                else:
                    V_X0 = self.model(x_0)
                
                V_candidate_prime = self.get_lyapunov_output(self.process_state(X_prime), x_0)

                # compute lie derivative using finite difference methods
                L_V = self.get_approx_lie_derivative(V_candidate, V_candidate_prime).squeeze()
                loss = self.lyapunov_loss(V_candidate, L_V, V_X0)
                circle_loss = 0.0
                if self.circle_tuning_loss is not None:
                    circle_loss = self.circle_tuning_loss(X, V_candidate)
                    loss += circle_loss
            
                total_loss += loss.item()
                total_V_loss += self.lyapunov_loss.V_loss
                total_lie_loss += self.lyapunov_loss.lie_loss
                total_eq_loss += self.lyapunov_loss.eq_loss
                total_circle_loss += circle_loss
                num_batches += 1
                loss.backward()
                self.optimizer.step() 


                #### FALSIFIER ####
                # run falsifier every falsifier_frequency epochs
                if (self.falsifier is not None) and epoch % (self.falsifier.get_frequency()) == 0:
                    num_batch_counterexamples = self.falsifier.add_counterexamples(X, sim_states, V_candidate, L_V)
                    num_counterexamples += num_batch_counterexamples

            # average losses for the epoch
            avg_loss = total_loss / num_batches
            avg_V_loss = total_V_loss / num_batches
            avg_lie_loss = total_lie_loss / num_batches
            avg_eq_loss = total_eq_loss / num_batches
            avg_circle_loss = total_circle_loss / num_batches
            loss_list.append(avg_loss)

            test_loss, test_V_loss, test_lie_loss, test_eq_loss, test_circle_loss = self.evaluate_test_set(test_dataloader, x_0)
            
            pbar.set_postfix({
                'Train Loss': f'{avg_loss:.4f}',
                'Test Loss': f'{test_loss:.4f}'
            })


            current_lr = self.optimizer.param_groups[0]['lr']
            log_dict = {
                "train/loss": avg_loss,
                "train/V_loss": avg_V_loss,
                "train/lie_loss": avg_lie_loss,
                "train/eq_loss": avg_eq_loss,
                "test/loss": test_loss,
                "test/V_loss": test_V_loss,
                "test/lie_loss": test_lie_loss,
                "test/eq_loss": test_eq_loss,
                "learning_rate": current_lr
            }
            
            # Add circle tuning loss if it's being used
            if self.circle_tuning_loss is not None:
                log_dict["train/circle_loss"] = avg_circle_loss
                log_dict["test/circle_loss"] = test_circle_loss
    
            if (self.falsifier is not None) and epoch % (self.falsifier.get_frequency()) == 0:
                if (num_counterexamples > 0): 
                    if verbose:
                        print("Not a Lyapunov function. Found {} counterexamples.".format(num_counterexamples))
                else:  
                    if verbose:
                        print('No counterexamples found!')
                    no_counterexamples_ct += 1
                    # end training early if no counterexamples are found 5 separate times
                    if no_counterexamples_ct == 5:
                        if verbose:
                            print("No counterexamples found for 5 iterations. Stopping early.")
                        break
                
                log_dict['falsifier/counterexamples'] = num_counterexamples

            wandb.log(log_dict, step=epoch)


        wandb.finish()

        return loss_list


class TrajectoryDataset(Dataset):
    def __init__(self, filename):
        # dataset of flattened tuples with dict-style observations
        loaded = self.load_dataset(filename)
        self.episode_lengths = torch.tensor(loaded['episode_lengths'], dtype=torch.int32)

        self.states_policy = torch.tensor(loaded['obs_policy'], dtype=torch.float32)
        self.states_critic = torch.tensor(loaded['obs_critic'], dtype=torch.float32)
        self.actions = torch.tensor(loaded['actions'], dtype=torch.float32)
        self.next_states_policy = torch.tensor(loaded['next_obs_policy'], dtype=torch.float32)
        self.next_states_critic = torch.tensor(loaded['next_obs_critic'], dtype=torch.float32)
        self.sim_state = loaded['sim_state']

    def load_dataset(self, filename):
        '''
        Loads named arrays from .npz and returns a dict
        '''
        loaded = np.load(filename, allow_pickle=True)
        return {key: loaded[key] for key in loaded}
    
    def __len__(self):
        return len(self.states_policy)

    def __getitem__(self, idx):
        policy = self.states_policy[idx]
        critic = self.states_critic[idx]
        next_policy = self.next_states_policy[idx]
        next_critic = self.next_states_critic[idx]

        obs = {'policy': policy, 'critic': critic}
        next_obs = {'policy': next_policy, 'critic': next_critic}
        sim_state = self.sim_state[idx]
        return obs, self.actions[idx], next_obs, sim_state

def collate_tensordict(batch):
    obs_list, actions_list, next_obs_list, sim_states = zip(*batch)
    obs_policy = torch.stack([b['policy'] for b in obs_list], dim=0)
    obs_critic = torch.stack([b['critic'] for b in obs_list], dim=0)
    next_policy = torch.stack([b['policy'] for b in next_obs_list], dim=0)
    next_critic = torch.stack([b['critic'] for b in next_obs_list], dim=0)
    actions = torch.stack(actions_list, dim=0)

    obs_td = TensorDict({'policy': obs_policy, 'critic': obs_critic}, batch_size=[obs_policy.shape[0]])
    next_td = TensorDict({'policy': next_policy, 'critic': next_critic}, batch_size=[next_policy.shape[0]])
    # sim_states is a tuple of dicts; return as list to keep objects intact
    return obs_td, actions, next_td, list(sim_states)