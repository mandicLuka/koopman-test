from dynamic_systems import DiscreteDynamicSystem
from dataclasses import dataclass, field
from typing import List, Tuple, Any
import numpy as np

@dataclass
class TrajectoryDescriptor:
    x_0: List[Any]
    dt_step: float
    start_time: float
    end_time: float

@dataclass
class Trajectory:
    description: TrajectoryDescriptor
    data: Any = field(default_factory=list)
    time: List[float] = field(default_factory=list)

class TrajectoryGenerator:

    def __init__(self, 
            dynamical_system: DiscreteDynamicSystem, 
            descriptors: List[TrajectoryDescriptor]):
        
        self._dynamical_system = dynamical_system
        self._dim = dynamical_system.dim
        self._descriptors = descriptors

    def evolve(self):

        trajectories = []
        # there can be multiple initial conditions
        # in the same scenario
        for t in self._descriptors:
            x_0 = t.x_0
            shape = x_0.shape
            dt = t.dt_step
            num_steps = int((t.end_time - t.start_time) / dt)
            x = np.zeros((shape[0], num_steps+1, *shape[1:]))
            x[:, 0, :] = x_0
            time = 0
            time_arr = [0] * (num_steps + 1)
            for i in range(num_steps):
                current_x = x[:, i, :]
                x[:, i+1] = self._dynamical_system(current_x, dt, time)
                time += dt
                time_arr[i+1] = time
            trajectories.append(Trajectory(t, x, time_arr))
        return trajectories

