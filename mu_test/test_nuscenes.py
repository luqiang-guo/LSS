from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

nusc = NuScenes(version='v1.0-mini', dataroot='/home/test/dataset/nuscenes/mini', verbose=True)
# print(nusc)
# print(nusc.list_scenes())
# print(dir(nusc))
scenes = create_splits_scenes()['mini_train']

samp = nusc.sample[0]
# print(samp)

{'token': 'ca9a282c9e77460f8360f564131a8af5', 
 'timestamp': 1532402927647951, 
 'prev': '', 
 'next': '39586f9d59004284a7114a68825e8eec', 
 'scene_token': 'cc8c0bf57f984915a77078b10eb33198', 
 'data': {'RADAR_FRONT': '37091c75b9704e0daa829ba56dfa0906', 
          'RADAR_FRONT_LEFT': '11946c1461d14016a322916157da3c7d', 
          'RADAR_FRONT_RIGHT': '491209956ee3435a9ec173dad3aaf58b', 
          'RADAR_BACK_LEFT': '312aa38d0e3e4f01b3124c523e6f9776', 
          'RADAR_BACK_RIGHT': '07b30d5eb6104e79be58eadf94382bc1', 
          'LIDAR_TOP': '9d9bf11fb0e144c8b446d54a8a00184f', 
          'CAM_FRONT': 'e3d495d4ac534d54b321f50006683844', 
          'CAM_FRONT_RIGHT': 'aac7867ebf4f446395d29fbd60b63b3b', 
          'CAM_BACK_RIGHT': '79dbb4460a6b40f49f9c150cb118247e', 
          'CAM_BACK': '03bea5763f0f4722933508d5999c5fd8', 
          'CAM_BACK_LEFT': '43893a033f9c46d4a51b5e08a67a1eb7', 
          'CAM_FRONT_LEFT': 'fe5422747a7d4268a4b07fc396707b23'}, 
          'anns': ['ef63a697930c4b20a6b9791f423351da', 
                   '6b89da9bf1f84fd6a5fbe1c3b236f809', 
                   '924ee6ac1fed440a9d9e3720aac635a0', 
                   '91e3608f55174a319246f361690906ba', 
                   '...',
                   '15a3b4d60b514db5a3468e2aef72a90c', 
                   '18cc2837f2b9457c80af0761a0b83ccc', 
                   '2bfcc693ae9946daba1d9f2724478fd4']}

# samples = [samp for samp in self.nusc.sample]


scene_name = nusc.get('scene', samp['scene_token'])['name']
print(scene_name)


samples = [samp for samp in nusc.sample]

# remove samples that aren't in this split
samples = [samp for samp in samples if nusc.get('scene', samp['scene_token'])['name'] in scenes]

# sort by scene, timestamp (only to make chronological viz easier)
samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))