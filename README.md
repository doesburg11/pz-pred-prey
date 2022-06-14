# PettingZoo predator prey AEC 
Reinforcement Learning in a Predator Prey model.
Basically an implementation with modifications of the paper:

“An Agent-Based Predator-Prey Model with Reinforcement Learning” by  M. Olsen and R. Fraczkowski.

### PettingZoo
This implementation does use the PettinZoo library for strictly AEC turned action behavior of the various agents (predators, Prey, Grass). This is also a workaround for unexpected behavior in the PettingZoo library with respect to the  removal of agents in strictly turn based agent selections. 

Behavior predators: 
- moving to Moore neighborhood (choice)
- eating prey (automatically)
- metabolism energy by aging (automatically)
- metabolism energy bij moving (choice)
- reproduce (automatically above threshold energy level)

### System Info
- Python 3.10
- Conda environment
- Linux Mint 20.3 Cinnamon
- IDE: DataSpell 2022.1

### References

- https://github.com/andrearama/PredatorPreyReinfLearning/
- [documents](docs/)
