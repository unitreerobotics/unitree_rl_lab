##
# Register Gym environments.
##

from isaaclab_tasks.utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = []
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
