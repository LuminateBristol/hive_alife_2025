from .bt_random_walk import bt as bt_random_walk
from .bt_logistics import logistics as logistics

behaviour_trees = {'random_walk': bt_random_walk,
                   'logistics': logistics}