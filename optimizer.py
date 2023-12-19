"""
File to contain all functions used to optimize the slicing
"""

import copy
import numpy as np
from typing import List

from geometry import Cut
from slicer import Slicer
from model import Model
import random



def calculate_weights(model: Model, slicing: Slicer) -> List[float]:
    """
    Calculate the amount of ur-realized potential shared surface area for each possible cut
    """
    weights = []
    for idx, cut in enumerate(slicing.cuts):
        if cut.is_cut:
            weights.append(-1)
        else:
            temp_model = copy.deepcopy(model)
            original_total_sa = 0
            new_total_sa = 0
            for other_cut in slicing.cuts:
                if other_cut.is_cut:
                    original_total_sa += 0
                else:
                    original_total_sa += other_cut.calculate_cut_surface_area(temp_model.mesh)
            temp_model.make_cut(cut)
            for other_idx, other_cut in enumerate(slicing.cuts):
                if other_cut.is_cut or other_idx == idx:
                    new_total_sa += 0
                else:
                    new_total_sa += other_cut.calculate_cut_surface_area(temp_model.mesh)

            weights.append(round(original_total_sa - new_total_sa, 4))

    return weights

def calculate_volume_weights(model: Model, slicing: Slicer) -> List[float]:
    weights = []
    for cut in slicing.cuts:
        if cut.is_cut:
            weights.append(-1)
        else:
            temp_model = copy.deepcopy(model)
            weights.append(cut.calculate_removed_volume(temp_model.mesh))

    return weights

def get_highest_weight_idx(weights: List[int], slicing: Slicer) -> int:
    """
    given a value array, give the idx of the best cut
    """
    max_value = 0
    best_idx = -1

    for idx,cut in enumerate(slicing.cuts):
        if not cut.is_cut:
            value = weights[idx]
            if value < 0:
                value = 0
            if value >= max_value:
                max_value = value
                best_idx = idx
    
    return best_idx

def give_next_cut_idx(model: Model, slicing: Slicer, method: str = "ordered") -> Cut:
    if method == "ordered":
        for idx,cut in enumerate(slicing.cuts):
            if not cut.is_cut:
                return idx
    
    if method == "optimal":
        weights = calculate_weights(model, slicing)
        return get_highest_weight_idx(weights, slicing)

    if method == "random":
        random.shuffle(slicing.cuts)
        for idx,cut in enumerate(slicing.cuts):
            if not cut.is_cut:
                return idx
            
    if method == "volume":
        weights = calculate_volume_weights(model, slicing)
        return get_highest_weight_idx(weights, slicing)

