#
##########################################################################################
## README
#
# This simple script can be used to convert single VHACD meshes into multiple geometries
# that can be used with MuJoCo. The input file can be any mesh that has already been split
# up into components (either with a VHACD tool or manually with Blender or similar), although
# it has been tested only with .obj files. The output will be a series of .stl files containing
# the mesh components, plus two .xml that can be read by MuJoCo. The .xml files can be imported
# into other MuJoCo files as: '<include file="my_complex_obj_geom.xml"/>'
#
##########################################################################################
#
## To use pymesh (via docker):
# docker pull pymesh/pymesh
#
## To run the docker file (change 'carlo' with any directory name that you need to access from the docker image):
# sudo docker run -it --mount type=bind,source=/home/carlo,target=/carlo -w /carlo pymesh/pymesh bash
#
## To cut objects in Blender
# 1) Press K, Z, C (C optional), then drag a perfect vertical line.
# 2) Menu: Select > Select Loop Inner Region, to select one side of the cut result (F6 > Select Bigger to select the other side).
# 3) In edit mode, press P, then Selected to create a new object with the current selection
#
## Useful links:
# https://pymesh.readthedocs.io/en/latest/installation.html#docker
# https://wiki.nexusmods.com/index.php/Splitting_meshes_in_Blender
# https://blender.stackexchange.com/a/7156
# http://www.mujoco.org/forum/index.php?threads/how-to-grasp-precisely-when-object-is-a-non-convex-hull.3917/
#


import pymesh
import numpy as np


def vhacd_to_mujoco(vhacd_file_path: str, output_dir: str, object_name: str,
					file_prefix: str=None, scale: float=1.0):

	file_prefix = file_prefix or object_name
	scale_str = " ".join([str(scale) for _ in range(3)])

	mesh_xml_f = '<mesh name="{mesh_name}" scale="{scale}" file="{file_name}"/>'
	geom_xml_f = '<geom type="mesh" mesh="{mesh_name}" rgba="{rgba}"/>'

	all_geom_xmls = []
	all_mesh_xmls = []
	
	mesh = pymesh.load_mesh(vhacd_file_path)
	components = pymesh.separate_mesh(mesh, connectivity_type='vertex')
	print(f'Found {len(components)} components.')

	for i, c in enumerate(components):
		mesh_name = f'{object_name}_part{i}'
		file_name = f'{file_prefix}_part{i}.stl'
		rgba = list(np.random.uniform(0.2, 1.0, 3)) + [1.0]
		rgba = " ".join([str(x) for x in rgba])
		mesh_xml = mesh_xml_f.format(mesh_name=mesh_name, scale=scale_str, file_name=file_name)
		geom_xml = geom_xml_f.format(mesh_name=mesh_name, rgba=rgba)
		all_mesh_xmls.append(mesh_xml)
		all_geom_xmls.append(geom_xml)
		file_path = output_dir + "/" + file_name
		pymesh.save_mesh(file_path, c)

	mj_geom_xml = '\n\t'.join(all_geom_xmls)
	mj_geom_xml = f"<mujoco>\n\t{mj_geom_xml}\n</mujoco>"

	mj_mesh_xml = '\n\t'.join(all_mesh_xmls)
	mj_mesh_xml = f"<mujoco>\n\t{mj_mesh_xml}\n</mujoco>"

	mj_geom_xml_path = output_dir + f"/{file_prefix}_geom.xml"
	with open(mj_geom_xml_path, 'w+') as f:
		f.write(mj_geom_xml)

	mj_mesh_xml_path = output_dir + f"/{file_prefix}_mesh.xml"
	with open(mj_mesh_xml_path, 'w+') as f:
		f.write(mj_mesh_xml)


if __name__ == '__main__':
	vhacd_to_mujoco(
		vhacd_file_path='./teapot_vhacd_manual.obj',
		output_dir='./',
		object_name='object_mesh:teapot_vhacd_m',
		file_prefix='teapot_vhacd_m',
	)
