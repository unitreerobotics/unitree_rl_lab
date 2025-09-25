import torch

class LyapunovRisk(torch.nn.Module):
    def __init__(self, lyapunov_factor=1., lie_factor=1., equilibrium_factor=1., 
                 lie_offset=0.1):
        super(LyapunovRisk, self).__init__()
        self.relu = torch.nn.ReLU()
        self.lyapunov_factor = lyapunov_factor
        self.lie_factor = lie_factor
        self.equilibrium_factor = equilibrium_factor
        self.lie_offset = lie_offset
        
        # Store individual loss components for logging
        self.V_loss = None
        self.lie_loss = None
        self.eq_loss = None

    def forward(self, V_candidate, L_V, V_X0):
        '''
        V_candidate: Candidate lyapunov function from model output
        L_V:         Lie derivative of lyapunov function (L_V = ∑∂V/∂xᵢ*fᵢ)
        V_X0:        Candidate lyapunov function evaluated at equilibrium conditions for state
        '''

        # lyapunov function must always be positive. Penalize negative outputs
        V_loss = self.relu(-V_candidate)
        # lie derivative must be negative 
        lie_loss = self.relu(L_V + self.lie_offset)
        # Lyapuvonv function evaluated at equilibrium points should be 0
        eq_loss = V_X0**2
        
        self.V_loss = V_loss.mean().item()
        self.lie_loss = lie_loss.mean().item()
        self.eq_loss = eq_loss.item()
        
        # weight loss factors individually
        total_risk = (self.lyapunov_factor*V_loss +  self.lie_factor*lie_loss).mean() + self.equilibrium_factor*eq_loss

        return total_risk

class CircleTuningLoss(torch.nn.Module):
    def __init__(self, state_max, tuning_factor=0.1):
        '''
        state_max: distance from lyapunov function.
                   Used to calculate normalized difference smaller.
        tuning_factor: weight for entire loss
        '''
        super(CircleTuningLoss, self).__init__()
        self.tuning_factor = tuning_factor
        self.state_max = state_max

    def forward(self, X, V_candidate):
        '''
        X: Input states
        V_candidate: Candidate lyapunov function from model output
        '''
        l2_norm = torch.sqrt(torch.sum(X**2, dim=1, keepdim=True))
        # distance of l2 norm of state from scaled lyapunov function 
        normalized_difference = l2_norm - self.state_max*V_candidate
        # regularize a circle for lyapunov loss
        circle_tuning_loss = self.tuning_factor * (normalized_difference**2)
        return circle_tuning_loss.mean()