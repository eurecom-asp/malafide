"""
Implementation of malafide filter. Malafilter?
This name is amazing.
"""

import torch
import pdb

class Malafide(torch.nn.Module):
    def __init__(self, filter_size,device, initial_dampening=0.7):
        super().__init__()
        assert filter_size % 2 == 1, f'Malafide filter size should be an odd number (got {filter_size})'

        self.naughty_filter = torch.nn.Conv1d(1, 1, filter_size, padding='same', bias=None)
        
        
        # initialize it as *almost* a dirac delta so that the initial attack is *almost* an identity
        # but first, dampen the standard initialization ever so slightly 
        self.naughty_filter.weight.data.mul_(initial_dampening)
        self.naughty_filter.weight.data[0, 0, filter_size//2] = 1

    def forward(self, x):
        return self.naughty_filter(x)

    def project(self, max_energy=None):
        """
        Apply projection contraints to the filter.
        This is basically PGD, but we are fancier and we say that 'the filter must have bounded energy'.
        Also, at every projection, the central spike of the filter is reset to 1 to forcefully preserve
            as much as possible from the original signal.
        
        Args:
            max_energy: project back to this energy. If None, will just reset the central spike to 1.
        """
        filter_size = self.naughty_filter.weight.numel()
        
        if max_energy is not None:
            print(f"Projecting to energy {max_energy}")
            # Don't count the energy of the central spike, we don't care about it
            current_energy = (torch.sum(self.naughty_filter.weight.detach()**2) - self.naughty_filter.weight.detach()[0,0,filter_size//2]**2).item()
            # compute the multiplicative coefficient for the projection
            # if the current energy is below the max, the coefficient is just 1
            # if the current energy is over the max, the vector is normalized to energy 1 then re-expanded to max energy
            projection_coeff = max_energy/max(current_energy, max_energy)
            self.naughty_filter.weight.data.mul_(projection_coeff)
        
        self.naughty_filter.weight.data[0, 0, filter_size//2] = 1 # set the middle spike to 1 again
        return self # return is technically not needed, but w/e

    def get_filter_size(self):
        return self.naughty_filter.weight.numel()
