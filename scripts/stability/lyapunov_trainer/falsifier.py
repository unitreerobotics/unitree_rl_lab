import torch
import numpy as np
import abc

class Falsifier(abc.ABC):
    def __init__(self, policy, obs_dim, action_dim, buffer_size, frequency=200, device='auto'):
        self.buffer = G1CounterExampleBuffer(policy, buffer_size, obs_dim, action_dim, device)
        self.counterexamples_added = 0
        self.frequency = frequency

    def get_frequency(self):
        return self.frequency

    def set_frequency(self, freq):
        self.frequency = freq

    @abc.abstractmethod
    def add_counterexamples(self, X, V_candidate, L_V):
        '''
        Finds current counterexamples by checking if the lyapunov conditions are violated for any sample. 
        counterexamples: all new examples from sampled from training data that don't satisfy the lyapunov conditions
        '''        
        pass
    

class SamplingBasedFalsifier(Falsifier):
    def __init__(self, policy, obs_dim, action_dim, lower_bound, upper_bound, epsilon=0., scale=0.02,
                 frequency=200, num_samples=10, device='auto'):
        super().__init__(policy, obs_dim, action_dim, frequency, device)

        self.epsilon = epsilon
        self.lower_bound = torch.Tensor(lower_bound).to(device)
        self.upper_bound = torch.Tensor(upper_bound).to(device)
        self.scale = scale

    @torch.no_grad
    def check_lyapunov(self, X, V_candidate, L_V):    
        '''
        Checks if the lyapunov conditions are violated for any sample. 
        Data points that are unsatisfiable will be sampled and added 
        '''
        N = X.shape[0]

        # Ensure lyapunov function and lie derivative are 1D tensors
        if V_candidate.dim() != 1:
            V_candidate = V_candidate.reshape(N)
        if L_V.dim() != 1:
            L_V = L_V.reshape(N)
        
        lyapunov_mask = (V_candidate < 0.)
        lie_mask = (L_V > self.epsilon)

        # bitwise or for falsification conditions
        union = lyapunov_mask.logical_or(lie_mask)

        # get batch indices that violate the lyapunov conditions
        indices = torch.nonzero(union).squeeze()
        # check num elements > 0
        if indices.numel() > 0:
            return  X[indices].reshape(indices.numel(), self.lower_bound.shape[0])
        else:
            return None
    
    @torch.no_grad
    def add_counterexamples(self, X, V_candidate, L_V):
        '''
        Take counter examples and sample points around them.
        X: current training data
        counterexamples: all new examples from training data that don't satisfy the lyapunov conditions
        ''' 
        counterexamples = self.check_lyapunov(X, V_candidate, L_V)   
        self.counterexamples_added += len(counterexamples) * self.num_samples
        for i in range(counterexamples.shape[0]):
            samples = torch.empty(self.num_samples, 0)
            counterexample = counterexamples[i]
            for j in range(self.upper_bound.shape[0]):
                lb = self.lower_bound[j]
                ub = self.upper_bound[j]
                value = counterexample[j]
                # Determine the min and max values for each feature in the chosen counterexamples
                min_value = torch.max(lb, value - self.scale * abs(value))
                max_value = torch.min(ub, value + self.scale * abs(value))
                
                sample = torch.Tensor(self.num_samples, 1).uniform_(min_value, max_value)
                samples = torch.cat([samples, sample], dim=1)
            self.buffer.add(samples)
        return X
    
class PGDFalsifier():
    '''
    Falsifier using projected gradient descent. Uses sign(grad) to push current inputs into regions where counterexamples can be found.
    
    '''
    def __init__(self, policy, obs_dim, action_dim, lower_bound, upper_bound, epsilon=0., sigma=1.0,
                 frequency=200, min_num_samples=10, device='auto'):
        super().__init__(policy, obs_dim, action_dim, frequency, device)

        self.epsilon = epsilon
        self.lower_bound = torch.Tensor(lower_bound).to(device)
        self.upper_bound = torch.Tensor(upper_bound).to(device)
        self.sigma = sigma

    def get_frequency(self):
        return self.frequency

    def set_frequency(self, freq):
        self.frequency = freq

    @torch.no_grad
    def check_lyapunov(self, X, V_candidate, L_V):    
        '''
        Checks if the lyapunov conditions are violated for any sample. 
        Data points that are unsatisfiable will be sampled and added 
        '''
        N = X.shape[0]

        # Ensure lyapunov function and lie derivative are 1D tensors
        if V_candidate.dim() != 1:
            V_candidate = V_candidate.reshape(N)
        if L_V.dim() != 1:
            L_V = L_V.reshape(N)
        
        lyapunov_mask = (V_candidate < 0.)
        lie_mask = (L_V > self.epsilon)

        # bitwise or for falsification conditions
        union = lyapunov_mask.logical_or(lie_mask)

        # get batch indices that violate the lyapunov conditions
        indices = torch.nonzero(union).squeeze()
        # check num elements > 0
        if indices.numel() > 0:
            return  X[indices].reshape(indices.numel(), self.lower_bound.shape[0])
        else:
            return None
    
    @torch.no_grad
    def add_counterexamples(self, X, V_candidate, L_V):
        '''
        Take counter examples and sample points around them.
        X: current training data
        counterexamples: all new examples from training data that don't satisfy the lyapunov conditions
        ''' 
        counterexamples = self.check_lyapunov(X, V_candidate, L_V)   
        self.counterexamples_added += len(counterexamples) * self.num_samples
        for i in range(counterexamples.shape[0]):
            samples = torch.empty(self.num_samples, 0)
            counterexample = counterexamples[i]
            for j in range(self.upper_bound.shape[0]):
                lb = self.lower_bound[j]
                ub = self.upper_bound[j]
                value = counterexample[j]
                # Determine the min and max values for each feature in the chosen counterexamples
                min_value = torch.max(lb, value - self.scale * abs(value))
                max_value = torch.min(ub, value + self.scale * abs(value))
                
                sample = torch.Tensor(self.num_samples, 1).uniform_(min_value, max_value)
                samples = torch.cat([samples, sample], dim=1)
            X = torch.cat((X, samples), dim=0)
        return X

class G1CounterExampleBuffer():
    """
    Stores (observations, actions, next_states).
    Observations are the counterexamples found. This is passed through the policy network to find next states. 
    """
    def __init__(self, env, policy, buffer_size=100000, obs_dim=None, action_dim=None, device="auto"):
        self.env = env
        self.policy = policy
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == "auto" else device
        
        if obs_dim is not None and action_dim is not None:
            self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
            self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
            self.next_observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
            
        self.pos = 0
        self.full = False
        self.size = 0
        

    def get_action(self, x):
        '''
        Take in single observation x to get action 
        '''
        # this RL model needs the previous action as input, so we store a class variable for action
        with torch.inference_mode():
            action = self.policy(x)
        return action

    def step(self, X, u):
        '''
        Generates all X_primes needed given current state and current action
        X: state
        u: action
        '''
        # take step in environment based upon current state and action
        N = X.shape[0]
        X_prime = torch.empty_like(X)
        for i in range(N):
            x_i = X[i, :].unsqueeze(0)
            reset_obs, _ = self.env.get_observations()

            # TODO set environment initial observation state as obs for reset instead of this reset
            current_sim_state = self.env.unwrapped.scene.get_state()
            current_robot_state = self.get_robot_state(self.process_state(x_i), self.process_state(reset_obs))
            current_sim_state['articulation']['robot'].update(current_robot_state)
            self.env.unwrapped.reset_to(current_sim_state, env_ids=None)
            # get current action to take 
            u_i = u[i, :].unsqueeze(0)
            # take step in environment
            x_prime, reward, terminate, info = self.env.step(u_i)
            # add sample to X_prime
            X_prime[i, :] = x_prime

        return X_prime
    
    def get_robot_state(self, obs, reset_obs):
        joint_pos_reset = reset_obs[:, :23]
        joint_vel_reset = reset_obs[:, 23:46]
        joint_pos = obs[:, :23] + joint_pos_reset
        joint_vel = obs[:, 23:46] + joint_vel_reset

        robot_state = {'joint_position' : joint_pos,
                       'joint_velocity' : joint_vel}
        return robot_state

    def add(self, observations):
        """
        Add (observation, action, next_observation) triples to the buffer.
        Must first pass through policy and environment to get the next actions and state.
        """

        actions = self.get_action(observations)
        next_observations = self.step(observations, actions)

        if len(observations) != len(actions):
            raise ValueError("Observations and actions must have the same batch size")
               
        # Add data to buffer
        for obs, action, next_obs in zip(observations, actions, next_observations):
            self.observations[self.pos] = obs
            self.actions[self.pos] = action
            self.next_observations[self.pos] = next_obs
            
            self.pos += 1
            self.size += 1
            
            # Wrap around when buffer is full
            if self.pos == self.buffer_size:
                self.full = True
                self.pos = 0
    
    def sample(self, batch_size):
        """
        Sample a batch of (observation, action, next_observation) triples.
        
        Args:
            batch_size: number of samples to return
            
        Returns:
            tuple: (observations, actions, next_observations) as numpy arrays
        """
        if self.size == 0:
            raise ValueError("Replay buffer is empty")
            
        # Sample indices
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        
        return (
            self.observations[batch_inds],
            self.actions[batch_inds], 
            self.next_observations[batch_inds]
        )
    
    def add_from_dataset(self, dataset):
        """
        Add all (observation, action, next_observation) triples from a dataset to the buffer.
        
        Args:
            dataset: PyTorch Dataset with __getitem__ returning (obs, action, next_obs)
        """
        # Collect all data first to batch efficiently
        observations = []
        actions = []
        next_observations = []
        
        for i in range(len(dataset)):
            item = dataset[i]
            
            obs, action, next_obs = item

            observations.append(obs.numpy())
            actions.append(action.numpy())
            next_observations.append(next_obs.numpy())
        

        observations = np.array(observations)
        actions = np.array(actions)
        next_observations = np.array(next_observations)
        
        self.add(observations, actions, next_observations)
    
    def to_torch(self, array, copy=True):
        """
        Convert numpy array to PyTorch tensor.
        
        Args:
            array: numpy array
            copy: whether to copy the data
            
        Returns:
            torch.Tensor
        """
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)
    
    def __len__(self):
        return self.size if not self.full else self.buffer_size

if __name__ == '__main__':
    # small test for falsifier
    num_samples=2
    falsifier = Falsifier(lower_bound=[-1., -0.5], upper_bound=[1., 0.5], num_samples=num_samples)
    x = torch.Tensor(5, 1).uniform_(-1., 1.)
    x = torch.cat((x, torch.Tensor(5, 1).uniform_(-0.5, 0.5)), dim=1)

    # Fake lyapunov funcions
    V = torch.ones(size=(5,1))
    V[3, 0] = -0.5
    V[2, 0] = -0.1

    L_V = -torch.ones(size=(5,1))
    L_V[1, 0] = 2.5

    # test that we find 3 unique counterexamples and add num_samples*3 to dataset
    counterexamples = falsifier.check_lyapunov(x, V, L_V)
    assert(counterexamples.size(0) == 3)
    x_new = falsifier.add_counterexamples(x, counterexamples)
    assert(x_new.size(0) == num_samples*counterexamples.size(0) + x.size(0))
    print(x)
    print(x_new)