from particle_graphics import *
import math
import copy
import time
import numpy as np
from scipy import sparse

SPACING = 0.01 #m spacing of particles
W_LENGTH = SPACING * 1.45
G_X = 0 #ms-2
G_Y = -10 #ms-2
KIN_VISCOSITY = 10**(-6) * 1 #m2s-1
RO_0 = 1000 #kgm-3
C0 = 1500 #ms-1
TIMESTEP = 0.000002 #s
FIELD_SIZE = 20 #number of fluid particles in y
WALL_THICKNESS = 3 #number of particles across wall
BOX_SIZE = W_LENGTH * 1.8214 #m gives 0.99 area of the weight function

def sort_to_boxes(x, y):
	box_count_x = int(max(x) / BOX_SIZE) + 2
	box_count_y = int(max(y) / BOX_SIZE) + 2
	boxes = []
	for i in range(box_count_x + 1):
		boxes.append([])
		for j in range(box_count_y + 1):
			boxes[-1].append([])	
	box_increments_x = np.arange(0, BOX_SIZE * (box_count_x + 3), BOX_SIZE)
	box_increments_y = np.arange(0, BOX_SIZE * (box_count_y + 3), BOX_SIZE)
	x_box_indexes = np.digitize(x, box_increments_x)
	y_box_indexes = np.digitize(y, box_increments_y)
	for i in range(len(x)):
		box_x = x_box_indexes[i]
		box_y = y_box_indexes[i]
		boxes[box_x][box_y].append(i)
	return np.array(boxes, dtype=object)
		
def get_friends(boxes):
	friends = np.roll(boxes, (0, -1), (0, 1))
	friends += np.roll(boxes, (-1, -1), (0, 1))
	friends += np.roll(boxes, (-1, 0), (0, 1))
	friends += np.roll(boxes, (-1, 1), (0, 1))
	return friends
	
def get_I(boxes, friends):
	boxes = boxes.flatten()
	friends = friends.flatten()
	boxes_lens = np.array([len(_) for _ in boxes])
	friends_lens = np.array([len(_) for _ in friends])
	i_list = []
	j_list = []
	for index in range(len(boxes)):
		j_list.extend(boxes_lens[index] * friends[index])
		for i in boxes[index]:
			i_list.extend([i] * friends_lens[index])
	for mybox in boxes:
		while len(mybox) > 0:
			i = mybox.pop(0)
			i_list.extend([i] * len(mybox))
			j_list.extend(mybox)
	rows = i_list #+ j_list
	cols = j_list #+ i_list
	double_rows = i_list + j_list
	double_cols = j_list + i_list
	rows = np.array(rows, dtype=np.int32)
	cols = np.array(cols, dtype=np.int32)
	double_rows = np.array(double_rows, dtype=np.int32)
	double_cols = np.array(double_cols, dtype=np.int32)
	return rows, cols, double_rows, double_cols

class Manager:
	def __init__(self):
		self.window = Tk()
		self.my_graphics = Graphics(self.window)
		self.iteration = 0
		self.time_start = time.time()
		self.time_running = 0
		self.step()
		
	def step(self):
		global x, y, u, v, ro, p, vol, new_u, new_v, new_ro, particle_type_list
		time_start = time.time()
		for i in range(100): #iterations per one graphics step
			### GETTING I INTERACTION MATRIX ROWS AND COLS
			boxes = sort_to_boxes(x, y)
			friends = get_friends(boxes)
			rows, cols, double_rows, double_cols = get_I(boxes, friends)
			### CALCULATING
			Dx = x[cols] - x[rows]
			Dy = y[cols] - y[rows]
			Du = u[cols] - u[rows]
			Dv = v[cols] - v[rows]
			Dro = ro[cols] - ro[rows]
			Dp = p[cols] + p[rows]
			Mr2 = Dx**2 + Dy**2
			#Wij, Wx, Wy
			Mexp = np.exp(-Mr2 / W_LENGTH ** 2)
			Wsize = (1 / (math.pi * (W_LENGTH ** 4))) * Mexp
			Wx = Dx * Wsize
			Wy = Dy * Wsize
			#Pi
			Pi = (8 * (Du * Dx + Dv * Dy)) / Mr2
			#DroDt
			Mro_data = Du * Wx + Dv * Wy
			Mro_data_double = np.concatenate((Mro_data, Mro_data))
			Mro = sparse.coo_matrix((Mro_data_double, (double_rows, double_cols)), shape=(particle_count, particle_count))
			DroDt = (-ro) * (Mro.dot(vol))
			#Diffusion term
			Mr = Mr2**0.5
			Mdiff_data = (Dro * (Dx * Wx + Dy * Wy)) / Mr
			Mdiff_data_double = np.concatenate((Mdiff_data, -Mdiff_data))
			Mdiff = sparse.coo_matrix((Mdiff_data_double, (double_rows, double_cols)), shape=(particle_count, particle_count))
			DroDt = DroDt + C0 * (Mdiff.dot(vol)) 
			#DuDt
			Mp_data = Dp * Wx
			Mp_data_double = np.concatenate((Mp_data, -Mp_data))
			Mp = sparse.coo_matrix((Mp_data_double, (double_rows, double_cols)), shape=(particle_count, particle_count))
			Mpi_data = Pi * Wx
			Mpi_data_double = np.concatenate((Mpi_data, -Mpi_data))
			Mpi = sparse.coo_matrix((Mpi_data_double, (double_rows, double_cols)), shape=(particle_count, particle_count))
			DuDt = (-1 / ro) * Mp.dot(vol)
			DuDt = DuDt + (KIN_VISCOSITY * RO_0 / ro) * (Mpi.dot(vol))
			DuDt = DuDt + np.full(particle_count, G_X)
			#DvDt
			Mp_data = Dp * Wy
			Mp_data_double = np.concatenate((Mp_data, -Mp_data))
			Mp = sparse.coo_matrix((Mp_data_double, (double_rows, double_cols)), shape=(particle_count, particle_count))
			Mpi_data = Pi * Wy
			Mpi_data_double = np.concatenate((Mpi_data, -Mpi_data))
			Mpi = sparse.coo_matrix((Mpi_data_double, (double_rows, double_cols)), shape=(particle_count, particle_count))
			DvDt = (-1 / ro) * Mp.dot(vol)
			DvDt = DvDt + (KIN_VISCOSITY * RO_0 / ro) * (Mpi.dot(vol))
			DvDt = DvDt + np.full(particle_count, G_Y)
			#adding additions to variables
			ro = ro + DroDt * TIMESTEP
			u = u + DuDt * TIMESTEP
			v = v + DvDt * TIMESTEP
			#recalculating pressure
			p = (C0 ** 2) * (ro - np.full(particle_count, RO_0))
			#enforcing zero velocity boundary conditions on wall
			u = u * particle_type_list
			v = v * particle_type_list
			#calculating new positions using new velocity
			x = x + u * TIMESTEP
			y = y + v * TIMESTEP
			
			self.iteration = self.iteration + 1
		self.time_running = self.time_running + (time.time() - time_start)
		print("Iterations finished: " + str(self.iteration) + " Simulation time: " + str(round(self.iteration * TIMESTEP, 6)) + "s")
		print("Average calculating time per iteration: " + str(self.time_running / self.iteration) + "s")
		print()
		self.my_graphics.update(x, y, p)
		self.window.after(1, self.step) #launch new iteration after 1ms (this is necessary for the graphics to work properly)

###START, generating particles
particle_count = (
(FIELD_SIZE + WALL_THICKNESS * 2) ** 2
-
0.5 * (FIELD_SIZE ** 2)
)
particle_count = int(round(particle_count))

print("Particle count:", particle_count)
print()

x = np.zeros(particle_count)
y = np.zeros(particle_count)
u = np.zeros(particle_count)
v = np.zeros(particle_count)
ro = np.full(particle_count, RO_0)
p = np.zeros(particle_count)
vol = np.full(particle_count, SPACING ** 2)
particle_type_list = np.zeros(particle_count) #0=wall, 1=fluid

### GENERATING PARTICLE COORDS

index = 0
### FLUID PARTICLE GENERATOR
for i in range(int(FIELD_SIZE / 2)):
	for j in range(FIELD_SIZE):
		x_coord = (WALL_THICKNESS + i) * SPACING
		y_coord = (WALL_THICKNESS + j) * SPACING
		x[index] = x_coord
		y[index] = y_coord
		particle_type_list[index] = 1
		index = index + 1
		
### WALL PARTICLE GENERATOR
for i in range(FIELD_SIZE + WALL_THICKNESS * 2):
	for j in range(FIELD_SIZE + WALL_THICKNESS * 2):
		overlap_x = max(
		[- i + WALL_THICKNESS, i - WALL_THICKNESS - FIELD_SIZE + 1]
		)
		overlap_y = max(
		[- j + WALL_THICKNESS, j - WALL_THICKNESS - FIELD_SIZE + 1]
		)
		overlap = max([overlap_x, overlap_y])   
		if overlap > 0:
			x_coord = i * SPACING
			y_coord = j * SPACING
			x[index] = x_coord
			y[index] = y_coord
			particle_type_list[index] = 0
			index = index + 1

mymanager = Manager()

mainloop()
