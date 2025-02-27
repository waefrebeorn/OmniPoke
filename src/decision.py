import utils

class Decision:
    """
    A simple class that delegates to Vision for the next action.
    This keeps the 'decision' logic in a separate file, but
    all the real work is done by Moondream in vision.py.
    """
    def __init__(self, vision):
        self.vision = vision

    def get_next_action(self):
        # Just call vision.get_next_action()
        action = self.vision.get_next_action()
        utils.log(f"Decision: {action}")
        return action
