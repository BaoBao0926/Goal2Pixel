from habitat import get_config

# from task_patch.ovon_register.config import HabitatConfigPlugin_ovon
# from task_patch.hm3d_register.config import HabitatConfigPlugin_hm3d
from task_patch.mp3d_register.config import HabitatConfigPlugin_mp3d
from task_patch.env_register.config import HabitatConfigPlugin_common

from habitat.config.default_structured_configs import register_hydra_plugin

def register_plugins():
    register_hydra_plugin(HabitatConfigPlugin_common)
    # register_hydra_plugin(HabitatConfigPlugin_ovon)
    # register_hydra_plugin(HabitatConfigPlugin_hm3d)
    register_hydra_plugin(HabitatConfigPlugin_mp3d)

if __name__ == "__main__":
    # Register custom hydra plugin
    register_plugins()

    # Get config
    config = get_config(config_path="configs/mp3d_config.yaml",
                        # overrides=[
                        #     "+habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map",
                        # ],
                        )

    breakpoint()