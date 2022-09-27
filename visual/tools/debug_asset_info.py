from isaacgym import gymapi


def run(asset_root, asset_file):
    gym = gymapi.acquire_gym()
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, gymapi.SimParams())

    asset = gym.load_asset(sim, asset_root, asset_file, gymapi.AssetOptions())
    gym.debug_print_asset(asset)


# gym.get_asset_rigid_shape_properties(asset)[0].friction


if __name__ == '__main__':
    asset_name = 'Kangaroo'
    asset_path = f'/home/cxl/aworkspace/codes/terrain-adaption/assets/{asset_name}'
    asset_file = f'{asset_name}.xml'
    run(asset_path, asset_file)
