import json
from pathlib import Path
def get_sensor_config(sensor_name):
    path = (Path(__file__).parent.parent / 'config' / 'sensor'/ f'{sensor_name}.json').resolve()
    assert path.exists(), f'Config file for {sensor_name} does not exist.'

    with open(path, 'r') as f:
        config = json.load(f)
    return config

def make_sensor(**kwargs):
    path = (Path(__file__).parent.parent / 'config' / 'sensor').resolve()
    with open(path / (kwargs['name'] + '.json'), 'w') as f:
        config = json.dump(kwargs, f, indent=4)

if __name__ == '__main__':
    make_sensor(
        name='multifunctional_v2',
        description='Multifunctional tactile sensor v2 used in Sajwani et al. 2023 TactiGraph',
        n_markers = 21,
        circle_radius = 85,
        center = (180, 117),
        frame_size = (346, 260)
    )
