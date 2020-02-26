import os
import copy
import yaml
import shutil
import trimesh
import numpy as np
import open3d as o3d
import os.path as osp
import transformations
from dm_control import mjcf
from attrdict import AttrDict
import PIL.Image as Image

from dm_control import viewer
import xml.etree.ElementTree as ET
from dm_control.suite import empty

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# add the mujoco_hand_exps utils path
curr_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(curr_dir)
mujoco_hand_exps_dir = os.path.join(base_dir, 'mujoco_hand_exps')
os.sys.path.append(mujoco_hand_exps_dir)
import utils

np.random.seed(1)


<camera pos="0 0.8 1." zaxis="0 0 1" name="yunchu_view" />
		if render:
img = self.sim.render(256, 256, camera_name="yunchu_view")
img = Image.fromarray(img)
img = img.rotate(-180)
img.save(f'{self.log_dir}/img_{self.another_timestep}.jpg')

class DatasetGenerator():
	def __init__(self, objects_file_path, radius_of_vis_cam, subsample_n, xml_skeleton,
		touch_traj_length=10, img_height=128, img_width=128, fovy=45.0,
		num_initial_touch_states=100, sensor_cam_fov=10.5, save_where=None):
		"""
		parameters:
		----------------
			objects_file_path: yaml config file for each of the object
			subsample_n: for how many objects you want to generate data for
			xml_skeleton: basic xml skeleton file on which I will load the object
			radius_of_vis_cam: what should be the radius of the circle in which the vis_cam moves
			touch_traj_length: How many depth images should be there in trajectory
			num_initial_touch_states : how many initial camera locations you want to sample
		"""
		if not osp.exists(objects_file_path):
			raise FileNotFoundError('No file with objects configurations found')

		# parse the yaml file of the objects
		with open(objects_file_path, 'r') as f:
			yaml_data = yaml.full_load(f)[0]

		self.EPS = 1e-6
		self.object_angle = yaml_data['angle']
		self.object_name = yaml_data['object_name']
		self.object_models = yaml_data['model_files']
		self.mesh_type = yaml_data['mesh_type']
		self.radius_of_vis_cam = radius_of_vis_cam
		self.img_height = img_height
		self.img_width = img_width
		self.fovy = fovy
		self.num_touch_cameras = num_initial_touch_states
		self.sensor_cam_fov = sensor_cam_fov
		if not save_where:
			raise AssertionError('save path is required')
		self.data_storage_path = save_where
		if not osp.exists(self.data_storage_path):
			os.makedirs(self.data_storage_path)
		self.obj_storage_path = osp.join(self.data_storage_path, self.object_name)
		if not osp.exists(self.obj_storage_path):
			os.makedirs(self.obj_storage_path)

		self.subsample_n = subsample_n
		if len(self.object_models) < subsample_n or subsample_n == -1:
			self.subsample_n = len(self.object_models)

		self.xml_skeleton = xml_skeleton
		self.touch_traj_length = touch_traj_length
		self.okay = False ## A flag which checks if the object is correctly loaded in mujoco

		## to be filled later ##
		self.object_name = None
		self.instance_storage_path = None
		self.valid_cups = []


	@staticmethod
	def get_object_name(object_file):
		splits = object_file.split('/')
		object_name = splits[-3]
		model_dir = '/'.join([p for p in splits[:-1]])
		return object_name, model_dir


	@staticmethod
	def scale_mesh(object_file):
		raise NotImplementedError

	@staticmethod
	def modify_and_change_xml(xml_string, stl_files_path, xml_save_path):
		with open(xml_save_path, 'w') as f:
			f.write(xml_string)

		tree = ET.parse(xml_save_path)
		root = tree.getroot()

		## need to write stuff again changing the paths of model files
		assets = root.findall('./asset')
		cnt = 0
		for m in assets[0].getchildren():
			if m.tag == 'mesh':
				m.attrib['file'] = osp.join(stl_files_path[cnt])
				m.attrib.pop('class')
				cnt += 1
		tree.write(xml_save_path)

	@staticmethod
	def get_obj_path(stl_file_path):
		"""
		Assumption is obj file, lies in this path and is named model_normalized.obj
		"""
		splits = stl_file_path.split('/')[:-2]
		base_path = '/'.join(s for s in splits)
		obj_model_fp = osp.join(base_path, 'model_normalized.obj')
		return obj_model_fp


	def _preprocess(self, object_file):
		"""
			object_file: path to obj model file
			Returns:
				name of the instance of the object, path_to_all_model_files
		"""
		# convert the mesh into stl format see it and then export
		if not osp.exists(object_file):
			raise FileNotFoundError('the specified mesh file is not found')
		object_name, model_dir = self.get_object_name(object_file)
		stl_models_path = osp.join(model_dir, 'stl_models')
		if not osp.exists(stl_models_path):
			os.makedirs(stl_models_path)

		mesh = trimesh.load_mesh(object_file)
		if not type(mesh) == trimesh.scene.scene.Scene:
			spath = os.path.join(stl_models_path, 'model_normalized_0.stl')
			mesh.export(spath)
		else:
			for i, m in enumerate(mesh.geometry):
				tmesh = mesh.geometry[m].copy()
				spath = osp.join(stl_models_path, 'model_normalized_{}.stl'.format(i))
				tmesh.export(spath)

		return object_name, stl_models_path


	def vis_data_collector(self, visual_mjcf, save_imgs):
		"""
			visual_mjcf : mjcf_file with object, this will be modified
			save_imgs : if True save_images
			Function:
				gets camera positions and quaternions, add them to mjcf
				for each cam_pos, and cam_quat, renders_imgs, computes their
				intrinsics and extrinsics and saves them instance_storage_path
		"""
		if save_imgs:
			color_img_dir = osp.join(self.instance_storage_path, 'images')
			if not osp.exists(color_img_dir):
				os.makedirs(color_img_dir)

		lookat_pos = visual_mjcf.worldbody.body['object:0'].pos
		center_of_movement = lookat_pos
		camera_positions, camera_quats = utils.generate_new_cameras(self.radius_of_vis_cam,
			center_of_movement, lookat_pos, height=0.0, jitter_z=True, jitter_amount=0.08)

		# TODO: Add a quiver plot indicating camera positions and quaternions

		ep_imgs = list()
		ep_depths = list()
		ep_intrinsics = list()
		ep_extrinsics = list()

		# now for each camera position and orientation get the image, depth, intrinsics and extrinsics
		for i, (pos, quat) in enumerate(zip(camera_positions, camera_quats)):
			print(f'generating for {i}/{len(camera_positions)} ...')
			# add the camera at this position with the given quaternion
			visual_mjcf.worldbody.add('camera', name=f'vis_cam:{i}', pos=pos, quat=quat, fovy=self.fovy)

			physics = mjcf.Physics.from_mjcf_model(visual_mjcf)
			img = physics.render(self.img_height, self.img_width, camera_id=f'vis_cam:{i}')
			depth = physics.render(self.img_height, self.img_width, camera_id=f'vis_cam:{i}', depth=True)

			assert img.shape[0] == self.img_height, "color img height is wrong"
			assert img.shape[1] == self.img_width, "color img width is wrong"
			assert depth.shape[0] == self.img_height, "depth img height is wrong"
			assert depth.shape[1] == self.img_width, "depth img width is wrong"

			if save_imgs:
				fig, ax = plt.subplots(2, sharex=True, sharey=True)
				ax[0].imshow(img)
				ax[1].imshow(depth)
				fig.savefig(f"{color_img_dir}/img_{i}.png")
				plt.close(fig=fig)

			# get the intrinsics and extrinsics and form the dict and store the data
			intrinsics = utils.dm_get_intrinsics(self.fovy, self.img_width, self.img_height)
			extrinsics = utils.dm_get_extrinsics(physics, physics.model.name2id(
				f'vis_cam:{i}', 'camera'
			))

			ep_imgs.append(np.flipud(img))
			ep_depths.append(np.flipud(depth))
			ep_intrinsics.append(intrinsics)
			ep_extrinsics.append(extrinsics)

		# get the data for the ref cam and you are done, first add the ref_cam
		visual_mjcf.worldbody.add('camera', name='ref_cam', pos=[0,-1,0], zaxis=[0,-1,0], fovy=self.fovy)

		physics = mjcf.Physics.from_mjcf_model(visual_mjcf)
		img = physics.render(self.img_height, self.img_width, camera_id='ref_cam')
		depth = physics.render(self.img_height, self.img_width, camera_id='ref_cam', depth=True)
		intrinsics = utils.dm_get_intrinsics(self.fovy, self.img_width, self.img_height)
		extrinsics = utils.dm_get_extrinsics(physics, physics.model.name2id(
			'ref_cam', 'camera'
		))

		if save_imgs:
			fig, ax = plt.subplots(2, sharex=True, sharey=True)
			ax[0].imshow(img)
			ax[1].imshow(depth)
			fig.savefig(f'{color_img_dir}/img_ref.png')
		ep_imgs.append(np.flipud(img))
		ep_depths.append(np.flipud(depth))
		ep_intrinsics.append(intrinsics)
		ep_extrinsics.append(extrinsics)

		# next recreate the scene !!
		if save_imgs:
			recon_imgs = utils.recreate_scene(ep_depths, ep_intrinsics, ep_extrinsics,
			camR_T_origin = np.linalg.inv(ep_extrinsics[-1]), clip_radius=5.0)

			for j, im in enumerate(recon_imgs):
				im = np.asarray(im)
				im = (im * 255.).astype(np.uint8)
				r_im = Image.fromarray(im)
				r_im.save(osp.join(self.instance_storage_path, f'visual_recon_img_{j}.png'))

		# create a dictionary to save the data and ship it !!
		save_dict = AttrDict()
		save_dict.rgb_camXs = np.stack(ep_imgs)
		save_dict.depth_camXs = np.stack(ep_depths)
		save_dict.intrinsics = np.stack(ep_intrinsics)
		save_dict.extrinsics = np.stack(ep_extrinsics)
		save_dict.camR_T_origin = np.linalg.inv(ep_extrinsics[-1])
		rgb_camRs = np.reshape(ep_imgs[-1], [1, self.img_height, self.img_width, 3])
		rgb_camRs = np.tile(rgb_camRs, [len(ep_imgs), 1, 1, 1])
		save_dict.rgb_camRs = rgb_camRs
		# everything should be len(51)
		for k in save_dict.keys():
			if k == 'camR_T_origin':
				continue
			assert len(save_dict[k]) == 51, "Data specific length is not right"
		return save_dict, visual_mjcf


	@staticmethod
	def sample_faces(vertices, faces, n_samples=5000):
		"""
		Samples point cloud on the surface of the model defined as vectices and
		faces. This function uses vectorized operations so fast at the cost of some
		memory.
		Parameters:
		vertices  - n x 3 matrix
		faces     - n x 3 matrix
		n_samples - positive integer
		Return:
		vertices - point cloud
		Reference :
		[1] Barycentric coordinate system
		\begin{align}
			P = (1 - \sqrt{r_1})A + \sqrt{r_1} (1 - r_2) B + \sqrt{r_1} r_2 C
		\end{align}
		"""
		actual_n_samples = n_samples
		vec_cross = np.cross(vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
							vertices[faces[:, 1], :] - vertices[faces[:, 2], :])
		face_areas = np.sqrt(np.sum(vec_cross ** 2, 1))
		face_areas = face_areas / np.sum(face_areas)

		# Sample exactly n_samples. First, oversample points and remove redundant
		# Error fix by Yangyan (yangyan.lee@gmail.com) 2017-Aug-7
		n_samples_per_face = np.ceil(n_samples * face_areas).astype(int)
		floor_num = np.sum(n_samples_per_face) - n_samples
		if floor_num > 0:
			indices = np.where(n_samples_per_face > 0)[0]
		floor_indices = np.random.choice(indices, floor_num, replace=True)
		n_samples_per_face[floor_indices] -= 1

		n_samples = np.sum(n_samples_per_face)

		# Create a vector that contains the face indices
		sample_face_idx = np.zeros((n_samples, ), dtype=int)
		acc = 0
		for face_idx, _n_sample in enumerate(n_samples_per_face):
			sample_face_idx[acc: acc + _n_sample] = face_idx
			acc += _n_sample

		r = np.random.rand(n_samples, 2);
		A = vertices[faces[sample_face_idx, 0], :]
		B = vertices[faces[sample_face_idx, 1], :]
		C = vertices[faces[sample_face_idx, 2], :]
		P = (1 - np.sqrt(r[:,0:1])) * A + np.sqrt(r[:,0:1]) * (1 - r[:,1:]) * B + \
			np.sqrt(r[:,0:1]) * r[:,1:] * C
		return P
	
	def normal_moving_strategy(self, mesh):
		"""
			mesh: actual mesh of the object, this should be trimesh.mesh object and not scene
			function:
				1. get uniformly sampled sampled points on the mesh using barycentric things
				2. compute the normals at each point
				3. The problem here is normals are not consistent meaning they can point in or out
				4. So I move in both the directions 0.07m.
				5. then compute the closest points on the mesh from these points
				6. Keep only the points which are 0.065 cm away from the mesh.
				7. rotate and return those
			returns:
				1. camera_locs, camera_orientations
		"""
		if not type(mesh) == trimesh.scene.scene.Scene:
			scene = trimesh.Scene(mesh)
		else:
			scene = mesh
		
		if not len(scene.geometry) == 1:
			return [-1], [-1]
		
		all_mesh_vertices = list()
		all_mesh_faces = list()
		for _, m in scene.geometry.items():
			all_mesh_vertices.append(m.vertices)
			all_mesh_faces.append(m.faces)
		
		all_mesh_vertices = np.concatenate(all_mesh_vertices, axis=0)
		all_mesh_faces = np.concatenate(all_mesh_faces, axis=0)

		sampled_points = self.sample_faces(all_mesh_vertices, all_mesh_faces)

		pcd_ = o3d.geometry.PointCloud()
		pcd_.points = o3d.utility.Vector3dVector(sampled_points)

		pcd_.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
		sampled_normals = np.asarray(pcd_.normals)

		# o3d.visualization.draw_geometries([pcd_])

		# move in both the directions
		moved_points_pos = sampled_points + 0.07*sampled_normals
		moved_points_neg = sampled_points + 0.07*(-sampled_normals)

		# compute the closest points from both the directions
		closest_pos, dists_pos, _ = mesh.nearest.on_surface(moved_points_pos)
		closest_neg, dists_neg, _ = mesh.nearest.on_surface(moved_points_neg)

		# now get the ok_idxs from both, ok idxs are defined as follows
		# if the distance is greater than 0.065 you are okay to me
		ok_idxs_pos = dists_pos >= 0.065
		print(ok_idxs_pos.sum())
		# you are the good points, as you are farther than 0.065
		ok_pos = moved_points_pos[ok_idxs_pos]
		# compute the directions for them
		corr_pos_pt_on_surface = closest_pos[ok_idxs_pos]
		assert len(ok_pos) == len(corr_pos_pt_on_surface), "these two should be equal"
		pos_direction = ok_pos - corr_pos_pt_on_surface
		pos_direction /= np.linalg.norm(pos_direction, axis=1).reshape(-1, 1)
		assert np.allclose(np.linalg.norm(pos_direction, axis=1), 1.0), "I normalized in the above line"

		ok_idxs_neg = dists_neg >= 0.065
		print(ok_idxs_neg.sum())
		# you are the good points, as you are farther than 0.065
		ok_neg = moved_points_neg[ok_idxs_neg]
		# compute the directions for the good points
		corr_neg_on_surface = closest_neg[ok_idxs_neg]
		assert len(ok_neg) == len(corr_neg_on_surface), "these should atleast have the same shape"
		neg_direction = ok_neg - corr_neg_on_surface
		neg_direction /= np.linalg.norm(neg_direction, axis=1).reshape(-1, 1)
		assert np.allclose(np.linalg.norm(neg_direction, axis=1), 1.0), "I normalized in the above line"

		# concatenate valid positives and negatives
		cam_locs = np.concatenate((ok_pos, ok_neg), axis=0)
		directions = np.concatenate((pos_direction, neg_direction), axis=0)

		# now draw the mesh and visualize, subsample rotate and return
		# ray_visualize = trimesh.load_path(np.hstack((cam_locs, cam_locs + 0.03*directions)).reshape(-1, 2, 3))
		# scene.add_geometry(ray_visualize)
		# scene.show()

		r_matrix = transformations.euler_matrix(*np.deg2rad(self.object_angle))
		cam_locs = np.c_[cam_locs, np.ones(len(cam_locs))]
		directions = np.c_[directions, np.ones(len(directions))]

		r_cam_locs = np.dot(r_matrix, cam_locs.T).T[:, :3]
		r_directions = np.dot(r_matrix, directions.T).T[:, :3]

		return r_cam_locs, r_directions

	
	def dont_care_strategy(self, mesh):
		if not type(mesh) == trimesh.scene.scene.Scene:
			scene = trimesh.Scene(mesh)
		else:
			scene = mesh
		
		if not len(scene.geometry) == 1:
			return [-1], [-1]
		
		all_mesh_vertices = list()
		all_mesh_faces = list()
		for _, m in scene.geometry.items():
			all_mesh_vertices.append(m.vertices)
			all_mesh_faces.append(m.faces)
		
		all_mesh_vertices = np.concatenate(all_mesh_vertices, axis=0)
		all_mesh_faces = np.concatenate(all_mesh_faces, axis=0)

		bbox_extent = mesh.bounding_box.bounds
		print(mesh.bounding_box.extents)
		# scale the bounding box
		new_bbox = np.copy(bbox_extent)
		new_bbox *= 1.5
		new_bbox_extent = new_bbox[1, :] - new_bbox[0, :]
		print(new_bbox_extent)
		# new_bbox_extent[0, 1] -= 0.05
		# new_extent = new_bbox_extent[1, :] - new_bbox_extent[0, :]
		# points_on_bbox = trimesh.sample.volume_rectangular(new_extent, 5000)

		longest_side = np.argmax(new_bbox_extent)
		lside_max, lside_min = new_bbox[1, longest_side], new_bbox[0, longest_side]

		# sample points in between these a lot of them
		sampled_points = np.random.uniform(lside_min, lside_max, size=30000)
		sampled_points = sampled_points.reshape(-1, 3)

		points_on_bbox = sampled_points
		pcd = trimesh.PointCloud(points_on_bbox)
		new_scene = trimesh.Scene([pcd, mesh])
		new_scene.show()

		location, dists, triangle_ids = mesh.nearest.on_surface(points_on_bbox)
		good_idxs = np.where(dists >= 0.07)

		selected_points = points_on_bbox[good_idxs]
		pcd = trimesh.PointCloud(selected_points)
		pcd.show()

		closest_locations, dist, triangle_id = mesh.nearest.on_surface(selected_points)
		directions = selected_points - closest_locations
		directions /= np.linalg.norm(directions, axis=1).reshape(-1, 1)
		print(directions)
		assert np.allclose(np.linalg.norm(directions, axis=1),1.0), "this should be 1"

		cam_locs = closest_locations + 0.07 * directions

		# now see here that all points are 7 cm away from the closest point on mesh
		# subsample them and return
		idxs = np.random.permutation(len(cam_locs))
		cam_locs = cam_locs[idxs[:2000]]
		directions = directions[idxs[:2000]]

		ray_visualize = trimesh.load_path(np.hstack((cam_locs, cam_locs + 0.03*directions)).reshape(-1, 2, 3))
		scene = trimesh.Scene([mesh, ray_visualize])
		scene.show()

		r_matrix = transformations.euler_matrix(*np.deg2rad(self.object_angle))
		cam_locs = np.c_[cam_locs, np.ones(len(cam_locs))]
		directions = np.c_[directions, np.ones(len(directions))]

		r_cam_locs = np.dot(r_matrix, cam_locs.T).T[:, :3]
		r_directions = np.dot(r_matrix, directions.T).T[:, :3]

		return r_cam_locs, r_directions

	
	def convex_sampling_strategy(self, mesh):
		if not type(mesh) == trimesh.scene.scene.Scene:
			mesh = trimesh.Scene(mesh)
		
		if not len(mesh.geometry) == 1:
			return [-1], [-1]
		
		all_mesh_vertices = list()
		all_mesh_faces = list()
		for _, m in mesh.geometry.items():
			all_mesh_vertices.append(m.vertices)
			all_mesh_faces.append(m.faces)
		
		all_mesh_vertices = np.concatenate(all_mesh_vertices, axis=0)
		all_mesh_faces = np.concatenate(all_mesh_faces, axis=0)

		# now fit the bounding box to the mesh and sample points
		points_on_bbox = mesh.bounding_box_oriented.sample_volume(count=8000)

		convex_hull_of_mesh = mesh.convex_hull
		# compute signed distance of all points on bounding box from convex_hull
		sdists = convex_hull_of_mesh.nearest.signed_distance(points_on_bbox)
		# get indexes which are greater than zero
		gidxs = np.where(sdists > 0.0)[0]
		filtered_points_on_bbox = points_on_bbox[gidxs]

		pcd = trimesh.PointCloud(filtered_points_on_bbox)
		pcd.show()

		# now where does each of the point intersect with the mesh
		closest_points_bbox, distances, triangle_id = convex_hull_of_mesh.nearest.on_surface(filtered_points_on_bbox)
		pcd_new = trimesh.PointCloud(closest_points_bbox)
		pcd_new.show()

		directions = closest_points_bbox - filtered_points_on_bbox
		directions /= np.linalg.norm(directions, axis=1).reshape(-1, 1)
		assert np.allclose(np.linalg.norm(directions, axis=1), 1.0), "directions are not normalized"

		cam_locs = closest_points_bbox + 0.05 * directions
		
		# finally check that all the points are outside the convex hull
		assert (convex_hull_of_mesh.nearest.signed_distance(cam_locs) < 0.0).all(), "some point is inside the convex hull of the object"

		# subsample and return
		idxs = np.random.permutation(len(cam_locs))
		idxs = idxs[:3000]
		cam_locs = cam_locs[idxs]
		directions = directions[idxs]

		ray_visualize = trimesh.load_path(np.hstack((cam_locs, cam_locs + 0.03*directions)).reshape(-1, 2, 3))
		scene = trimesh.Scene([convex_hull_of_mesh, ray_visualize])
		scene.show()

		# rotate the points and directions now as required by the mujoco
		r_matrix = transformations.euler_matrix(*np.deg2rad(self.object_angle))
		cam_locs = np.c_[cam_locs, np.ones(len(cam_locs))]
		directions = np.c_[directions, np.ones(len(directions))]

		r_cam_locs = np.dot(r_matrix, cam_locs.T).T[:, :3]
		r_directions = np.dot(r_matrix, directions.T).T[:, :3]
			
		return r_cam_locs, r_directions
	
	@staticmethod
	def mynormalize(arr):
		arr /= np.linalg.norm(arr, axis=1).reshape(-1, 1)
		assert np.allclose((np.linalg.norm(arr, axis=1) - 1), 0), "directions are not normalized"
		return arr

	def get_close_cam_pos_and_quats(self, mesh):
		if not type(mesh) == trimesh.scene.scene.Scene:
			mesh = trimesh.Scene(mesh)
		if not len(mesh.geometry) == 1:
			return [-1], [-1]
		all_mesh_vertices = list()
		all_mesh_faces = list()
		for _, m in mesh.geometry.items():
			all_mesh_vertices.append(m.vertices)
			all_mesh_faces.append(m.faces)
		
		all_mesh_vertices = np.concatenate(all_mesh_vertices, axis=0)
		all_mesh_faces = np.concatenate(all_mesh_faces, axis=0)

		uniform_pts = self.sample_faces(all_mesh_vertices, all_mesh_faces, n_samples=5000)

		# compute the normals
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(uniform_pts)
		# compute the normal at each point
		pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
			radius=0.1, max_nn=30))
		normals = np.asarray(pcd.normals)

		# move in the normal directions a little bit
		normal_dir_moved_points = uniform_pts + 0.09 * normals
		pcd = trimesh.PointCloud(normal_dir_moved_points)
		pcd.show()

		# NOTE that the points are inside and outside since the mesh direction is not consistently pointing outwards
		# NOTE there is no functionality as of now to do that in open3d good opportunity to do code contribution

		# check that none of the points intersect with the mesh
		# if the points lie on the mesh then the distance to the closest point would be zero
		# also I know I will be working with one mesh only
		mesh_elem = list(mesh.geometry.keys())
		mesh_elem = mesh.geometry[mesh_elem[0]]
		mesh_elem.show()

		filter_idxs = list()
		for _, m in mesh.geometry.items():
			closest_points_bbox, distances, triangle_id = m.nearest.on_surface(normal_dir_moved_points)
			if (distances - 0.0 <= self.EPS).any():
				filter_idxs.append(np.where((distances + self.EPS) >= 0)[0])
			assert not np.allclose(distances, 0.0), "some point lies very close to the mesh not good"
		
		if len(filter_idxs) > 0:
			filter_idxs = np.asarray(filter_idxs).flatten()
			uniform_pts = uniform_pts[filter_idxs]
			normal_dir_moved_points = normal_dir_moved_points[filter_idxs]
		
		pcd_new = trimesh.PointCloud(uniform_pts)
		pcd_new.show()
		
		directions = normal_dir_moved_points - uniform_pts
		directions = self.mynormalize(directions)

		cam_locs = uniform_pts + 0.08*directions
		pcd_newer = trimesh.PointCloud(cam_locs)
		pcd_newer.show()

		# subsample
		idxs = np.random.permutation(len(cam_locs))
		idxs = idxs[:990]

		cam_locs = cam_locs[idxs]
		directions = directions[idxs]

		# filter out the ones which intersect with the mesh
		for _, m in mesh.geometry.items():
			closest_pts, distances, triangle_id = m.nearest.on_surface(cam_locs)
			if (distances - 0.0 <= self.EPS).any():
				# means one of the camera_location is close to the object, filter it out
				idxs = np.where((distances - 0.0) <= self.EPS)
				print('these are the indexes where distance is equal to zero')
				from IPython import embed; embed()
				mask = np.ones(len(cam_locs))
				mask[idxs] = 0
				cam_locs = cam_locs[mask.astype(bool)]
				directions = directions[mask.astype(bool)]

		# rotate the points and directions now as required by the mujoco
		r_matrix = transformations.euler_matrix(*np.deg2rad(self.object_angle))
		cam_locs = np.c_[cam_locs, np.ones(len(cam_locs))]
		directions = np.c_[directions, np.ones(len(directions))]

		r_cam_locs = np.dot(r_matrix, cam_locs.T).T[:, :3]
		r_directions = np.dot(r_matrix, directions.T).T[:, :3]

		ray_visualize = trimesh.load_path(np.hstack((cam_locs[:, :3], cam_locs [:, :3] + 0.05*directions[:, :3])).reshape(-1, 2, 3))
		mesh.add_geometry(ray_visualize)
		mesh.show()

		print(f'final number of cameras returned: {len(cam_locs)}')

		return r_cam_locs, r_directions

	def touch_data_collector(self, mesh_obj, touch_mjcf, save_imgs, debug=False):
		"""
			parameters:
			-------------
				mesh_obj : file_path to stl files
				touch_mjcf : the basic mjcf with the object loaded up
				save_imgs : if true save the intermediate images
				mesh_files : stl files path for scene files.
				debug : if True adds a phantom object to the mjcf and hence to xml
			function:
			--------------
				find the bounding box for the object, search the mujoco functionality for this.
				sample points on the surface of the obtained box, these are the camera locations
				collect intrinsics, extrinsics, depth and color patches at all these locations
			algorithm:
			--------------
			1. first find the minimum enclosing sphere in the form of C and radius from the mesh
			2. Generate points on the surface of the sphere of the radius and center specified.
			3. filter points based on the height.
			4. Collect data using this
		"""
		if save_imgs:
			sensor_img_path = osp.join(self.instance_storage_path, "sensor_imgs")
			if not osp.exists(sensor_img_path):
				os.makedirs(sensor_img_path)

		# I am making a dataspecific assumption here, to get the model_obj file
		mesh_obj_fp = self.get_obj_path(mesh_obj[0])
		if not osp.exists(mesh_obj_fp):
			raise FileNotFoundError

		# load the file
		mesh = trimesh.load(mesh_obj_fp)
		# once you loaded the mesh do the processing to get the camera locations and camera orientations
		if self.mesh_type == "convex":
			close_up_cam_locs, close_up_cam_quats = self.convex_sampling_strategy(mesh)
		elif self.mesh_type == "non_convex":
			close_up_cam_locs, close_up_cam_quats = self.normal_moving_strategy(mesh)
		else:
			raise ValueError('you have to specify the mesh type')
		# close_up_cam_locs, close_up_cam_quats = self.new_strategy(mesh)
		# close_up_cam_pos, close_up_cam_quats, scene_image = self.get_close_cam_pos_and_quats(mesh)
		if len(close_up_cam_locs) == 1 and len(close_up_cam_quats) == 1:
			return [-1], [-1], [-1], [-1], [-1]

		# if here means everything went well and we are ready to collect data
		chosen_cam_locs = close_up_cam_locs
		to_keep_idxs, cam_quats = utils.get_quaternion(close_up_cam_quats, world_up=[0, 0, 1.])
		chosen_cam_locs = chosen_cam_locs[to_keep_idxs.astype(bool)]
		assert len(chosen_cam_locs) == len(cam_quats), "some filtering is wrong"
		# pcd = o3d.geometry.PointCloud()
		# pcd.points = o3d.utility.Vector3dVector(chosen_cam_locs)
		# o3d.visualization.draw_geometries([pcd])
		
		# here I have got all the camera positions and orientation with all the filtering required
		sensor_imgs = list()
		sensor_depths = list()
		sensor_intrinsics = list()
		sensor_extrinsics = list()

		# now instead of adding all the cameras to mjcf I will just add one camera and change its location
		# and orientation before the start of every run. Also I will not reconstruct the shape of the object
		# that can be put to later processing.
		camera_placeholder = touch_mjcf.worldbody.add('camera', name='sensor_camera',
			pos=[0, 0, 0], quat=[0, 0, 0, 1], fovy=self.sensor_cam_fov)

		# collect data
		for i, (cam_loc, cam_quat) in enumerate(zip(chosen_cam_locs, cam_quats)):
			# if i == 10:
			#     break
			# change the position and location of the sensor camera based on the computed pos and quat
			camera_placeholder.pos = cam_loc
			camera_placeholder.quat = cam_quat
			print(f'--- generating for {i}/{len(chosen_cam_locs)} ---')
			
			if debug:
				# add the cameras at those position and orientation too
				touch_mjcf.worldbody.add('camera', name=f'sensor_cam:{i}', pos=cam_loc, quat=cam_quat,
					fovy=self.sensor_cam_fov)

			# add a phantom body here for debugging
			if debug:
				phantom_body = touch_mjcf.worldbody.add('body', name=f'cam_pos:{i}', pos=cam_loc, quat=cam_quat)
				phantom_body.add('geom', name=f'cam_pos:{i}', type='sphere', size=[0.01], contype=0, conaffinity=0)

			physics = mjcf.Physics.from_mjcf_model(touch_mjcf)
			sensor_img = physics.render(self.img_height, self.img_width, camera_id='sensor_camera')
			sensor_depth = physics.render(self.img_height, self.img_width, camera_id='sensor_camera', depth=True)

			assert sensor_img.shape[0] == self.img_height, "color img height is wrong"
			assert sensor_img.shape[1] == self.img_width, "color img width is wrong"
			assert sensor_depth.shape[0] == self.img_height, "depth img height is wrong"
			assert sensor_depth.shape[1] == self.img_width, "depth img width is wrong"

			if save_imgs:
				fig, ax = plt.subplots(2, sharex=True, sharey=True)
				ax[0].imshow(sensor_img)
				ax[1].imshow(sensor_depth)
				fig.savefig(f"{sensor_img_path}/img_{i}.png")
				plt.close(fig=fig)

			# get the intrinsics and extrinsics and form the dict and store the data
			intrinsics = utils.dm_get_intrinsics(self.sensor_cam_fov, self.img_height, self.img_width)
			extrinsics = utils.dm_get_extrinsics(physics, physics.model.name2id(
				'sensor_camera', 'camera'
			))
			sensor_imgs.append(np.flipud(sensor_img))
			sensor_depths.append(np.flipud(sensor_depth))
			sensor_intrinsics.append(intrinsics)
			sensor_extrinsics.append(extrinsics)

		sensor_imgs = np.stack(sensor_imgs)
		sensor_depths = np.stack(sensor_depths)
		sensor_intrinsics = np.stack(sensor_intrinsics)
		sensor_extrinsics = np.stack(sensor_extrinsics)

		# assert that the length of all four arrays is the same
		length_of_data = len(sensor_imgs)
		assert len(sensor_depths) == length_of_data, "some data is missing"
		assert len(sensor_intrinsics) == length_of_data, "some intrinsics are missing"
		assert len(sensor_extrinsics) == length_of_data, "some extrinsics are missing"

		camR_T_origin = np.asarray([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00],
									[ 0.00000000e+00,  2.22044605e-16,  1.00000000e+00, 2.22044605e-16],
									[ 0.00000000e+00,  1.00000000e+00, -2.22044605e-16, 1.00000000e+00],
									[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])

		# # recreate the scene using the computed intrinsics, extrinsics and depths
		recon_imgs = None
		if save_imgs:
			recon_imgs = utils.recreate_scene(sensor_depths, sensor_intrinsics, sensor_extrinsics,
				camR_T_origin=camR_T_origin , clip_radius=5.0)

		save_dict = AttrDict()
		save_dict.sensor_imgs = sensor_imgs
		save_dict.sensor_depths = sensor_depths
		save_dict.sensor_intrinsics = sensor_intrinsics
		save_dict.sensor_extrinsics = sensor_extrinsics
		save_dict.camR_T_origin = camR_T_origin

		return save_dict, touch_mjcf, recon_imgs, chosen_cam_locs, cam_quats


	def create_env_and_collect_data(self, object_file, save_imgs,
		collect_visual_data, collect_touch_trajectories):
		"""
		Create the new environment with the specified object from xml_skeleton file
		object_file: path to the model file
		save_imgs : if True save the intermediate images from each view
		collect_visual_data : if True copies the data_mjcf to visual_mjcf and collects
			vis data
		collect_touch_data  : if True copies the data_mjcf to touch_mjcf and collects
			touch data
		Function:
			saves the visual data, and the visual mjcf by converting it to xml
			saves the touch data, and the touch mjcf by converting it to xml
		"""

		self.okay = True
		self.object_name, stl_models_path = self._preprocess(object_file)

		stl_files_path = [osp.join(stl_models_path, stl_file) for stl_file in os.listdir(stl_models_path)]
		if len(stl_files_path) > 1:
			# that means the mesh is broken into different elements I am ignoring such meshes for now
			return

		data_mjcf = mjcf.from_path(self.xml_skeleton)
		for i, stl_file in enumerate(stl_files_path):
			data_mjcf.asset.add('mesh', name=f'mesh:{i}', file=stl_file)
			object_body = data_mjcf.worldbody.add('body', name=f'object:{i}', pos=[0, 0, 0], euler=self.object_angle)
			object_body.add('geom', type='mesh', name=f'object:{i}', mesh=f'mesh:{i}',\
				group='1', condim='3', mass='1000')
		
		## this is temporary for my heart satisfaction
		# temp_body = data_mjcf.worldbody.add('body', name=f'object:100', pos=[0.4, 0.0, 0.4], quat=[0, 0, 0, 1])
		# temp_body.add('geom', type='sphere', name=f'object_geom:100', size='0.3', group='1', contype='0', conaffinity='0', mass='1000')

		# make the environment with empty file as of now
		self.modify_and_change_xml(data_mjcf.to_xml_string(), stl_files_path,
			xml_save_path='/home/bbgsp/anaconda3/envs/muj_exps/lib/python3.6/site-packages/dm_control/suite/changed.xml')
		task_kwargs = dict(xml='/home/bbgsp/anaconda3/envs/muj_exps/lib/python3.6/site-packages/dm_control/suite/changed.xml')

		# since I filtered out the invalid ones the next is not needed anymore and self.okay should start with true
		# def loader():
		#     env = empty.SUITE['empty'](**task_kwargs)
		#     return env

		# try:
		#     viewer.launch(loader)
		#     self.okay = True
		#     self.valid_cups.append(object_file)
		# except Exception as inst:
		#     print('got an exception')\
		## .. not needed end .. ##

		# this creates the directory of the object based on pre-eliminary check of it being loadable
		if self.okay:
			# meaning I will want to collect data for this
			self.instance_storage_path = osp.join(self.obj_storage_path, self.object_name)
			if not osp.exists(self.instance_storage_path):
				os.makedirs(self.instance_storage_path)
		
		if self.okay and collect_touch_trajectories:
			print('---- COLLECTING TOUCH DATA ----')
			touch_mjcf = copy.deepcopy(data_mjcf)
			# touch_data, modified_touch_mjcf = self.touch_data_collector(stl_files_path, touch_mjcf, save_imgs)
			touch_data, modified_touch_mjcf, recon_imgs, chosen_cam_locs, chosen_cam_quats =\
				self.touch_data_collector(stl_files_path, touch_mjcf, save_imgs)
			if len(chosen_cam_locs) == 1 and len(chosen_cam_quats) == 1:
				self.okay = False
			
			# if self.okay is false I will remove the above created directory
			if not self.okay:
				shutil.rmtree(self.instance_storage_path)
			
			if self.okay:
				touch_save_path = osp.join(self.instance_storage_path, "touch_data.npy")
				np.save(touch_save_path, touch_data)
				# save the modified xml too, NOTE not useful anymore since only one camera is modified over and over
				touch_xml_string = modified_touch_mjcf.to_xml_string()
				xml_save_path = osp.join(self.instance_storage_path, "touch_data_xml.xml")
				self.modify_and_change_xml(touch_xml_string, stl_files_path, xml_save_path)
				# save the reconstructed images
				# save the PIL images of the reconstruction
				# again for speed this would almost always be none
				if recon_imgs:
					for j, im in enumerate(recon_imgs):
						im = np.asarray(im)
						im = (im * 255.).astype(np.uint8)
						r_img = Image.fromarray(im)
						r_img.save(osp.join(self.instance_storage_path, f"recon_img_{j}.png"))
				# save the camera positions and quaternions
				cam_pos_save_path = osp.join(self.instance_storage_path, "cam_positions.npy")
				np.save(cam_pos_save_path, chosen_cam_locs)
				cam_quats_save_path = osp.join(self.instance_storage_path, "cam_quat.npy")
				np.save(cam_quats_save_path, chosen_cam_quats)


		if self.okay and collect_visual_data:
			# meaning the file loaded properly and everything is smooth till now
			# collect the visual data
			# now from the center of the object which is at 0,0,0 I want the camera to be 1m apart in the xy plane

			visual_mjcf = copy.deepcopy(data_mjcf)
			vis_data, modified_vis_mjcf = self.vis_data_collector(visual_mjcf, save_imgs)

			vis_save_path = osp.join(self.instance_storage_path, "visual_data.npy")
			np.save(vis_save_path, vis_data)

			# save the xml too
			vis_xml_save_path = osp.join(self.instance_storage_path, "object_mjcf.xml")
			self.modify_and_change_xml(modified_vis_mjcf.to_xml_string(), stl_files_path, vis_xml_save_path)


	def collect_data(self, save_imgs=False, collect_visual_data=False, collect_touch_data=False):
		# sample the indices
		idxs = np.random.permutation(len(self.object_models))
		idxs = idxs[:self.subsample_n]
		idx = 0
		cnt = 0
		while True:
			print(f'--- collecting {cnt}/{self.subsample_n} datapoint ---')
			if cnt == self.subsample_n:
				break
			chosen_yaml = self.object_models[idx]
			self.create_env_and_collect_data(chosen_yaml, save_imgs=save_imgs,
				collect_visual_data=collect_visual_data, collect_touch_trajectories=collect_touch_data)
			idx += 1
			if self.okay:
				cnt += 1
		print("done")
		from IPython import embed; embed()


if __name__ == '__main__':
	data_gen = DatasetGenerator('resources/cups.yaml', radius_of_vis_cam=1.0, subsample_n=10,
		xml_skeleton='resources/xml_skeleton.xml', save_where="/home/bbgsp/Documents/Projects/mujoco_hand_exps/trajectory_env/new_close_up_dataset")
	data_gen.collect_data(save_imgs=False, collect_visual_data=True, collect_touch_data=True)
