from enum import Enum
from .cache_tree import SequenceStateNode


class EliminationState(Enum):
    SHARED = 1
    ALLOCATED = 2
    CANDIDATE_FOR_DELETION = 3
    UNALLOCATED = 4


class EliminationProtocol:
    def __init__(self, el:EliminationState):
        self.current_state = EliminationState.UNALLOCATED
        return
 
    def next_state(node: SequenceStateNode, top_of_eviction_queue):
        current_state, ref_count, address  = node.state, node.ref_count, node.address 
        next_state = current_state
        if current_state == EliminationState.ALLOCATED:
            if ref_count < 1:
                next_state = EliminationState.CANDIDATE_FOR_DELETION
            elif ref_count > 1:
                next_state = EliminationState.SHARED
        elif current_state == EliminationState.SHARED:
            if ref_count == 1:
                next_state = EliminationState.ALLOCATED
        elif current_state == EliminationState.CANDIDATE_FOR_DELETION:
            if ref_count == 1:
                next_state = EliminationState.ALLOCATED
            elif ref_count == 0 and top_of_eviction_queue == address:
                next_state = EliminationState.UNALLOCATED
        elif current_state == EliminationState.UNALLOCATED:
            if ref_count > 0:
                next_state = EliminationState.ALLOCATED
        return next_state


