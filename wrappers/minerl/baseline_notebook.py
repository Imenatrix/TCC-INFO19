import numpy as np

def dataset_action_batch_to_actions(dataset_actions, camera_margin=3):
    """
    Turn a batch of actions from dataset (`batch_iter`) to a numpy
    array that corresponds to batch of actions of ActionShaping wrapper (_actions).

    Camera margin sets the threshold what is considered "moving camera".

    Note: Hardcoded to work for actions in ActionShaping._actions, with "intuitive"
        ordering of actions.
        If you change ActionShaping._actions, remember to change this!

    Array elements are integers corresponding to actions, or "-1"
    for actions that did not have any corresponding discrete match.
    """
    # There are dummy dimensions of shape one
    camera_actions = dataset_actions["camera"].squeeze()
    attack_actions = dataset_actions["attack"].squeeze()
    forward_actions = dataset_actions["forward"].squeeze()
    jump_actions = dataset_actions["jump"].squeeze()
    batch_size = len(camera_actions)
    actions = np.zeros((batch_size,), dtype=int)

    for i in range(len(camera_actions)):
        # Moving camera is most important (horizontal first)
        if camera_actions[i][0] < -camera_margin:
            actions[i] = 4
        elif camera_actions[i][0] > camera_margin:
            actions[i] = 5
        elif camera_actions[i][1] > camera_margin:
            actions[i] = 6
        elif camera_actions[i][1] < -camera_margin:
            actions[i] = 7
        elif forward_actions[i] == 1:
            if jump_actions[i] == 1:
                actions[i] = 3
            else:
                actions[i] = 2
        elif attack_actions[i] == 1:
            actions[i] = 1
        else:
            # No reasonable mapping (would be no-op)
            actions[i] = 0
    return actions