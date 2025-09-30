import torch
import numpy as np
import abc
from tensordict import TensorDict

class Falsifier(abc.ABC):
    def __init__(self, env, policy, buffer_size=100000, lyapunov_by_construction=False, frequency=20, device='auto'):

        self.buffer = G1CounterExampleBuffer(env, policy, buffer_size, device)
        self.counterexamples_added = 0
        self.frequency = frequency
        self.lyapunov_by_construction = lyapunov_by_construction

    def get_frequency(self):
        return self.frequency

    def set_frequency(self, freq):
        self.frequency = freq

    @abc.abstractmethod
    def add_counterexamples(self, X, sim_states, V_candidate, L_V):    
        pass
    
    def get_robot_state(self, obs):
        # relative joint pos and vel
        joint_pos_rel = self.get_joint_pos(obs)
        joint_vel = self.get_joint_vel(obs)

        default_joint_pos = self.env.unwrapped.scene["robot"].data.default_joint_pos[0]
        joint_pos = joint_pos_rel + default_joint_pos

        root_velocity = torch.zeros(1, 6)
        root_velocity[:, 3:6] = self.get_base_ang_vel(obs)
        # initial pose for unitree g1
        root_pose = torch.tensor([[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        robot_state = {
            'root_pose': root_pose,
            'root_velocity': root_velocity,
            'joint_position': joint_pos.unsqueeze(0),
            'joint_velocity': joint_vel.unsqueeze(0)
        }
        return robot_state


class SamplingBasedFalsifier(Falsifier):
    def __init__(self, env, policy, lower_bound, upper_bound, epsilon=0., scale: torch.Tensor =0.02, num_samples=10, 
                 buffer_size=1000000, lyapunov_by_construction=False, frequency=20, device='auto'):
        super().__init__(env, policy, buffer_size, lyapunov_by_construction, frequency, device)
        self.env = env
        self.epsilon = epsilon
        self.lower_bound = torch.Tensor(lower_bound).to(device)
        self.upper_bound = torch.Tensor(upper_bound).to(device)
        self.scale = scale
        self.num_samples = num_samples

    @torch.no_grad
    def check_lyapunov(self, X, V_candidate, L_V):    
        '''
        Checks if the lyapunov conditions are violated for any sample. 
        Data points that are unsatisfiable will be sampled and added 
        '''
        N = X.shape[0]


        # Ensure lyapunov function and lie derivative are 1D tensors

        if L_V.dim() != 1:
            L_V = L_V.reshape(N)
        
        lie_mask = (L_V > -self.epsilon)
        if self.lyapunov_by_construction:
            # only need to check lie derivative for counterexamples
            indices = torch.nonzero(lie_mask).squeeze()
        else:
            if V_candidate.dim() != 1:
                V_candidate = V_candidate.reshape(N)
            lyapunov_mask = (V_candidate < 0.)

            # bitwise or for falsification conditions
            union = lyapunov_mask.logical_or(lie_mask)
            # get batch indices that violate the lyapunov conditions as well
            indices = torch.nonzero(union).squeeze()

        return indices
    
    @torch.no_grad
    def add_counterexamples(self, X, sim_states, V_candidate, L_V):
        '''
        Finds current counterexamples by checking if the lyapunov conditions are violated for any sample.
        Slightly perturbs counterexamples and adds to the buffer.
        ''' 
        indices = self.check_lyapunov(X, V_candidate, L_V)
        if indices.numel() == 0:
            return 0

        if indices.dim() == 0:
            indices = indices.unsqueeze(0)
        
        counterexamples = X[indices]
        sim_state_ces = [sim_states[i] for i in indices.tolist()]

        obs_policy_counterexamples = counterexamples['policy']
        obs_critic_counterexamples = counterexamples['critic']

        

        min_values = torch.max(self.lower_bound, 
                              obs_policy_counterexamples - self.scale * torch.abs(obs_policy_counterexamples))
        max_values = torch.min(self.upper_bound, 
                              obs_policy_counterexamples + self.scale * torch.abs(obs_policy_counterexamples))
        
        min_values_expanded = min_values.unsqueeze(1).expand(-1, self.num_samples, -1)
        max_values_expanded = max_values.unsqueeze(1).expand(-1, self.num_samples, -1)
        
        # sample uniformly 
        rand_samples = torch.rand_like(min_values_expanded)
        samples = min_values_expanded + rand_samples * (max_values_expanded - min_values_expanded)
        
        # Reshape to (num_counterexamples * num_samples, feature_dim)
        samples_flat = samples.reshape(-1, samples.shape[-1])
        self.counterexamples_added += samples_flat.shape[0]

        critic_first_15 = obs_critic_counterexamples[:, :15]
        critic_first_15_flat = critic_first_15.repeat_interleave(self.num_samples, dim=0)
        sample_obs_critic = torch.cat([critic_first_15_flat, samples_flat], dim=1)
        sample_observations = TensorDict({
            'policy': samples_flat,
            'critic': sample_obs_critic
        }, batch_size=[samples_flat.shape[0]], device=samples_flat.device)
        
        # Create sim_state_samples with updated robot states from perturbed observations
        sim_state_samples = []
        for i in range(len(sim_state_ces)):
            # Get the original sim_state for this counterexample
            original_sim_state = sim_state_ces[i]
            
            # For each sample generated from this counterexample
            for j in range(self.num_samples):

                sample_idx = i * self.num_samples + j
                perturbed_obs = sample_observations[sample_idx]
                
                perturbed_policy_tensor = perturbed_obs['policy'].unsqueeze(0) 
                updated_robot_state = self.buffer.get_robot_state(perturbed_policy_tensor)
                
                # Keep the original root pose and root velocity, only update joint states
                original_robot_state = original_sim_state['articulation']['robot']
                # Convert numpy arrays to tensors on the correct device
                root_pose_tensor = torch.tensor(original_robot_state['root_pose'], device=self.env.device)
                root_velocity_tensor = torch.tensor(original_robot_state['root_velocity'], device=self.env.device)
                

                joint_pos = updated_robot_state['joint_position']
                joint_vel = updated_robot_state['joint_velocity']
                
                final_robot_state = {
                    'root_pose': root_pose_tensor,
                    'root_velocity': root_velocity_tensor,
                    'joint_position': joint_pos,
                    'joint_velocity': joint_vel
                }
                
                updated_sim_state = original_sim_state.copy()

                updated_sim_state['articulation']['robot'] = final_robot_state
                
                sim_state_samples.append(updated_sim_state)
        
        # Add to buffer
        self.buffer.add(sample_observations, sim_state_samples)
        return sample_observations.shape[0]


class PGDFalsifier(Falsifier):
    '''
    Falsifier using projected gradient descent. Uses sign(grad) to push current inputs into regions where counterexamples can be found.
    
    '''
    def __init__(self, env, policy, lower_bound, upper_bound, epsilon=0., sigma=1.0,
                 min_num_samples=10, buffer_size=1000000, lyapunov_by_construction=False, frequency=200, device='auto'):
        super().__init__(env, policy, buffer_size, lyapunov_by_construction, frequency, device)

        self.epsilon = epsilon
        self.lower_bound = torch.Tensor(lower_bound).to(device)
        self.upper_bound = torch.Tensor(upper_bound).to(device)
        self.sigma = sigma

    
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
    buffer that stores (obs_td, actions, next_obs_td, sim_states).
    - obs_td and next_obs_td are TensorDicts with keys 'policy' and 'critic'.
    - actions is a torch tensor of shape (N, action_dim).
    - sim_states is a list of dict
    """
    def __init__(self, env, policy, buffer_size:int=100000, device="auto"):
        self.env = env
        self.policy = policy
        self.capacity = buffer_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == "auto" else device

        self.observations: TensorDict | None = None
        self.next_observations: TensorDict | None = None
        self.actions: torch.Tensor | None = None
        self.sim_states: list | None = None

        self.ptr = 0
        self.size = 0

    def get_joint_pos(self, X):
        joint_pos_start = 45 + 4*23
        joint_pos_end = joint_pos_start + 23
        joint_pos_rel = X[:, joint_pos_start:joint_pos_end]
        return joint_pos_rel

    def get_joint_vel(self, X):
        '''
        Get joint velocity from observation and unnormalize
        '''
        joint_pos_start = 45 + 4*23
        joint_pos_end = joint_pos_start + 23
        joint_vel_start = joint_pos_end + 4*23
        joint_vel_end = joint_vel_start + 23
        joint_vel_rel = X[:, joint_vel_start:joint_vel_end]
        return joint_vel_rel / 0.05

    def get_base_ang_vel(self, X):
        '''
        Get base angular velocity from observation and unnormalize
        '''
        return X[:, 12:15] / 0.2

    def get_robot_state(self, obs):
        # relative joint pos and vel
        joint_pos_rel = self.get_joint_pos(obs)
        joint_vel = self.get_joint_vel(obs)

        default_joint_pos = self.env.unwrapped.scene["robot"].data.default_joint_pos[0]
        joint_pos = joint_pos_rel + default_joint_pos

        # Get device from obs tensor
        
        root_velocity = torch.zeros(1, 6, device=self.device)

        root_velocity[:, 3:6] = self.get_base_ang_vel(obs)
        # initial pose for unitree g1
        root_pose = torch.tensor([[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=self.device)
        robot_state = {
            'root_pose': root_pose,
            'root_velocity': root_velocity,
            'joint_position': joint_pos.unsqueeze(0),
            'joint_velocity': joint_vel.unsqueeze(0)
        }
        return robot_state

    def get_action(self, x):
        '''
        Take in single observation x to get action 
        '''
        # this RL model needs the previous action as input, so we store a class variable for action
        with torch.inference_mode():
            action = self.policy(x)
        return action

    def fill_history(self, next_obs, obs):
        previous_indices_list = (
            list(range(3, 15)) +       # base_ang_vel
            list(range(18, 30)) +      # projected_gravity
            list(range(33, 45)) +      # velocity_commands
            list(range(68, 160)) +     # joint_pos_rel
            list(range(183, 275)) +    # joint_vel_rel
            list(range(298, 390))      # last_action
        )

        history_indices_list = (
            list(range(0, 12)) +       # base_ang_vel
            list(range(15, 27)) +      # projected_gravity
            list(range(30, 42)) +      # velocity_commands
            list(range(45, 137)) +     # joint_pos_rel
            list(range(160, 252)) +    # joint_vel_rel
            list(range(275, 367))      # last_action
        )
        previous_indices_policy = torch.tensor(previous_indices_list, dtype=torch.int64)
        history_indices_policy = torch.tensor(history_indices_list, dtype=torch.int64)

        previous_indices_list_critic = (
            list(range(3, 15)) +       # base_lin_vel
            list(range(18, 30)) +      # base_ang_vel
            list(range(33, 45)) +      # projected_gravity
            list(range(48, 60)) +      # velocity_commands
            list(range(83, 175)) +     # joint_pos_rel
            list(range(198, 290)) +    # joint_vel_rel
            list(range(313, 405))      # last_action
        )

        history_indices_list_critic = (
            list(range(0, 12)) +       # base_lin_vel
            list(range(15, 27)) +      # base_ang_vel
            list(range(30, 42)) +      # projected_gravity
            list(range(45, 57)) +      # velocity_commands
            list(range(60, 152)) +     # joint_pos_rel
            list(range(175, 267)) +    # joint_vel_rel
            list(range(290, 382))      # last_action
        )
        previous_indices_critic = torch.tensor(previous_indices_list_critic, dtype=torch.int64)
        history_indices_critic = torch.tensor(history_indices_list_critic, dtype=torch.int64)

        next_obs['policy'][:, history_indices_policy] = obs['policy'][:, previous_indices_policy]
        next_obs['critic'][:, history_indices_critic] = obs['critic'][:, previous_indices_critic]
        return next_obs

    def step(self, observations, actions, sim_states):
        '''
        Step the simulator for each (obs, action, sim_state) to get next_obs TensorDicts.
        '''
        N = actions.shape[0]
        next_obs_list = []
        for i in range(N):
            sim_state = sim_states[i]

            self.env.unwrapped.reset_to(sim_state, env_ids=None)

            u_i = actions[i].unsqueeze(0)
            next_obs, reward, terminate, info = self.env.step(u_i)
            next_obs = self.fill_history(next_obs, observations[i].unsqueeze(0))
            next_obs_list.append(next_obs[0])
        return TensorDict.stack(next_obs_list, dim=0)

    def add(self, observations: TensorDict, sim_states: list):
        """
        Add counterexamples by computing actions with the policy and stepping the simulator.
        Required:
        - observations: TensorDict with keys 'policy' and 'critic' (batch first)
        - sim_states_batch: list of simulator states (len == batch)
        """
        N = observations.batch_size[0] if isinstance(observations.batch_size, tuple) else observations.shape[0]
        if len(sim_states) != N:
            raise ValueError("sim_states_batch length must match batch size of observations")
        
        actions = self.get_action(observations)

        # Step simulator to get next observations
        next_observations= self.step(observations, actions, sim_states)

        # initialize storages
        if self.observations is None:
            policy_dim = observations['policy'].shape[1]
            critic_dim = observations['critic'].shape[1]
            self.observations = TensorDict({
                'policy': torch.zeros(self.capacity, policy_dim, dtype=torch.float32, device=self.device),
                'critic': torch.zeros(self.capacity, critic_dim, dtype=torch.float32, device=self.device),
            }, batch_size=[self.capacity], device=self.device)
            self.next_observations = TensorDict({
                'policy': torch.zeros(self.capacity, policy_dim, dtype=torch.float32, device=self.device),
                'critic': torch.zeros(self.capacity, critic_dim, dtype=torch.float32, device=self.device),
            }, batch_size=[self.capacity], device=self.device)
            self.actions = torch.zeros(self.capacity, actions.shape[1], dtype=torch.float32, device=self.device)
            self.sim_states = [None] * self.capacity

        # Write by index with wrap-around
        idx = torch.arange(self.ptr, self.ptr + N, device=self.device) % self.capacity
        self.observations['policy'][idx] = observations['policy']
        self.observations['critic'][idx] = observations['critic']
        self.actions[idx] = actions
        self.next_observations['policy'][idx] = next_observations['policy']
        self.next_observations['critic'][idx] = next_observations['critic']
        for k, i in enumerate(idx.tolist()):
            self.sim_states[i] = sim_states[k]

        self.ptr = int((self.ptr + N) % self.capacity)
        self.size = min(self.size + N, self.capacity)
    
    def sample(self, batch_size):
        """
        Sample a batch of (obs_td, actions, next_obs_td, sim_states).
        Returns obs_td and next_obs_td as TensorDicts stacked on batch dimension.
        """
        if self.size == 0:
            raise ValueError("Replay buffer is empty")
        upper = self.capacity if self.size == self.capacity else self.ptr
        idx = torch.randint(0, upper, (batch_size,), device=self.device)
        obs_td = TensorDict({
            'policy': self.observations['policy'][idx],
            'critic': self.observations['critic'][idx],
        }, batch_size=[batch_size], device=self.device)
        next_obs_td = TensorDict({
            'policy': self.next_observations['policy'][idx],
            'critic': self.next_observations['critic'][idx],
        }, batch_size=[batch_size], device=self.device)
        actions = self.actions[idx]
        sim_states = [self.sim_states[i.item()] for i in idx]
        return obs_td, actions, next_obs_td, sim_states
    
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
        return self.size
