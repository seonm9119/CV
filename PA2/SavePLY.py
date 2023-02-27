import numpy as np

def SavePLY(filename, X):

	ply_header = 'ply\n'
	ply_header += 'format ascii 1.0\n'
	ply_header += 'element vertex %(vert_num)d\n'
	ply_header += 'property float x\n'
	ply_header += 'property float y\n'
	ply_header += 'property float z\n'
	ply_header += 'property uchar diffuse_red\n'
	ply_header += 'property uchar diffuse_green\n'
	ply_header += 'property uchar diffuse_blue\n'
	ply_header += 'end_header\n'

	with open(filename, 'w') as f:
		f.write(ply_header%dict(vert_num=len(X)))
		np.savetxt(f,X,'%f %f %f %d %d %d')




