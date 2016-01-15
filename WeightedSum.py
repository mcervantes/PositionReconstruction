import numpy as np
from pax import plugin


class PosRecWeightedSum(plugin.PosRecPlugin):
    """Reconstruct x,y positions as the charge-weighted average of PMT positions in the top array.
    """
    def reconstruct_position(self, peak):
        hitpattern = peak.area_per_channel[self.pmts]
        return np.average(self.pmt_locations, weights=hitpattern, axis=0)


class PosRecScaledWeightedSum(plugin.PosRecPlugin):
    """Modified version of PosRecWeightedSum. The positon calculated by scaling the x-y calculated by PosRecWeightedSum and constrainint it to 
       to be inside of the TPC
    """
    def reconstruct_position(self, peak):
        hitpattern = peak.area_per_channel[self.pmts]
        pos = np.average(self.pmt_locations,
                          weights=hitpattern,
                          axis=0)
        scale = 1.085 
        rad = (pos[0]**2+pos[1]**2)**(1/2)
        if rad*scale < self.config["tpc_radius"]:
            return pos*scale
        else:
            return pos

         
