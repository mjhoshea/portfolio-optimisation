from collections import namedtuple

Transition = namedtuple('transitions', ['state', 'action', 'reward', 'next_state', 'is_done'])