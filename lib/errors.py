from typing import Iterable


class ApprovalCommitteeError(Exception):
    def __init__(self, index: Iterable[str]):
        assert len(index) == 3,\
            "Should be a triplet (proposer, current_state, next_state)"
        self.index = index
    def __str__(self):
        error_msg = (
            "The following transition could not be "
            "handled: Proposer: {}, from state "
            "{} to {}.").format(*self.index)
        return error_msg