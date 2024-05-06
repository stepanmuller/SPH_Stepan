from particle_graphics import *
import math
import copy
import time
import numpy as np

SPACING = 0.01 #m spacing of particles
W_LENGTH = SPACING * 1.45
G_X = 0 #ms-2
G_Y = -10 #ms-2
KIN_VISCOSITY = 10**(-6) * 1 #m2s-1
RO_0 = 1000 #kgm-3
C0 = 1500 #ms-1
TIMESTEP = 0.000003 #s
FIELD_SIZE = 20 #number of fluid particles in y
WALL_THICKNESS = 3 #number of particles across wall
BOX_SIZE = W_LENGTH * 1.8214 #m gives 0.99 area of the weight function

def add_ro_u_v(i, j_list): #adds contribution to Dro/Dt, Du/Dt, Dv/Dt of 
	#all j to i and of i to all j
	global x_list, y_list, u_list, v_list, ro_list, p_list, vol_list, new_u_list, new_v_list, new_ro_list, particle_type_list
	x1 = x_list[i]
	y1 = y_list[i]
	u1 = u_list[i]
	v1 = v_list[i]
	ro1 = ro_list[i]
	p1 = p_list[i]
	vol1 = vol_list[i]
	particle_type1 = particle_type_list[i]
	for j in j_list: #calculating pairs of i with all j
		x2 = x_list[j]
		y2 = y_list[j]
		
		dx = x2 - x1
		dy = y2 - y1
		norm = dx ** 2 + dy ** 2
		
		if norm ** 0.5 < (W_LENGTH * 1.8214):		
			
			u2 = u_list[j]
			v2 = v_list[j]
			ro2 = ro_list[j]
			p2 = p_list[j]
			vol2 = vol_list[j]
			particle_type2 = particle_type_list[j]
		
			du = u2 - u1
			dv = v2 - v1
			dro = ro2 - ro1
			
			if norm == 0:
				norm = 10 ** (-6)
			
			#calculating wij gradient
			wij_size = (
			2 * math.e ** ( - (dx**2 + dy**2) / W_LENGTH ** 2)
			/
			(math.pi * W_LENGTH ** 4)
			)
			grad_1 = dx * wij_size
			grad_2 = dy * wij_size
			#calculating contribution to Dro/Dt

			K1_ro = du * grad_1 +  dv * grad_2 #divergence constant for ro
			K2_ro = C0 * dro * (dx * grad_1 + dy * grad_2) / norm ** 0.5 #diffusion constant for ro
			addition_ro_to_i = - ro1 * vol2 * K1_ro + vol2 * K2_ro
			addition_ro_to_j = - ro2 * vol1 * K1_ro + vol2 * ( - K2_ro)
			new_ro_list[i] = new_ro_list[i] + addition_ro_to_i
			new_ro_list[j] = new_ro_list[j] + addition_ro_to_j
				
			#calculating contribution to Du/Dt and Dv/Dt
			pi = 8 * (du * dx + dv * dy) / norm #viscious term constant
				
			K1_u = (p2 + p1) * grad_1 #pressure term constants
			K2_u = KIN_VISCOSITY * RO_0 * pi * grad_1 #viscious term constants
			K1_v = (p2 + p1) * grad_2  
			K2_v = KIN_VISCOSITY * RO_0 * pi * grad_2
				
			addition_u_to_i = (vol2 / ro1) * ((- K1_u) + K2_u)
			addition_u_to_j = (vol2 / ro1) * (K1_u + (- K2_u))
			addition_v_to_i = (vol2 / ro1) * ((- K1_v) + K2_v)
			addition_v_to_j = (vol2 / ro1) * (K1_v + (- K2_v))
			
			if particle_type1 == 0:
				new_u_list[i] = new_u_list[i] + addition_u_to_i
				new_v_list[i] = new_v_list[i] + addition_v_to_i
			if particle_type2 == 0:
				new_u_list[j] = new_u_list[j] + addition_u_to_j
				new_v_list[j] = new_v_list[j] + addition_v_to_j

def update_pressure(i): #updates pressure of i based on new ro
	global p_list, new_ro_list
	p_list[i] = C0 ** 2 * (new_ro_list[i] - RO_0)

def update_position(i): #updates x and y from new velocity
	global x_list, y_list, new_u_list, new_v_list
	avg_u = new_u_list[i]
	avg_v = new_v_list[i]
	x_list[i] = x_list[i] + avg_u * TIMESTEP
	y_list[i] = y_list[i] + avg_v * TIMESTEP

class Manager:
	def __init__(self):
		self.window = Tk()
		self.my_graphics = Graphics(self.window)
		self.iteration = 0
		self.time_start = time.time()
		self.time_running = 0
		self.step()
		
	def step(self):
		global x_list, y_list, u_list, v_list, ro_list, p_list, vol_list, new_u_list, new_v_list, new_ro_list, particle_type_list
		time_start = time.time()
		### SORTING FLUID PARTICLES INTO BOXES EACH ITERATION
		box_list = copy.deepcopy(wall_boxes)
		for i in range(len(x_list)):
			if particle_type_list[i] == 0:
				box_x = int(x_list[i] / BOX_SIZE)
				box_y = int(y_list[i] / BOX_SIZE)
				box_list[box_x][box_y].append(i)
		### LOOPING OVER BOXES
		for box_x in range(len(box_list)):
			for box_y in range(len(box_list[box_x])):
				box = box_list[box_x][box_y]
				neighbours = []
				for i in [0, 1]: #ADDING NEIGHBOURS
					#we don't need to add -1 because we solved those particles before
					if box_x + i >= 0 and box_x + i < len(box_list):
						for j in [-1, 0, 1]:
							if box_y + j >= 0 and box_y + j < len(box_list[box_x]):
								if abs(i) + abs(j) != 0:
									neighbours.extend(box_list[box_x + i][box_y + j])
				while len(box) > 0:
					i = box.pop(0)
					j_list = box + neighbours
					add_ro_u_v(i, j_list)
		
		new_ro_list = new_ro_list * TIMESTEP #new ro
		new_ro_list = copy.deepcopy(new_ro_list) + copy.deepcopy(ro_list)
		for i in range(len(x_list)):
			update_pressure(i)
			if particle_type_list[i] == 0: #if fluid
				new_u_list[i] = (new_u_list[i] + G_X) * TIMESTEP #new u
				new_v_list[i] = (new_v_list[i] + G_Y) * TIMESTEP #new v
				new_u_list[i] = new_u_list[i] + u_list[i]
				new_v_list[i] = new_v_list[i] + v_list[i]
				update_position(i)

		u_list = copy.deepcopy(new_u_list)
		v_list = copy.deepcopy(new_v_list)
		new_u_list = np.zeros(particle_count)
		new_v_list = np.zeros(particle_count)
		new_ro_list = np.zeros(particle_count)
			
		self.time_running = self.time_running + (time.time() - time_start)
		
		self.iteration = self.iteration + 1
		if self.iteration % 100 == 0:
			print("Iterations finished: " + str(self.iteration) + " Simulation time: " + str(round(self.iteration * TIMESTEP, 6)) + "s")
			print("Average calculating time per iteration: " + str(self.time_running / self.iteration) + "s")
			print()
			self.my_graphics.update(x_list, y_list, p_list)
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

#particle_list = np.zeros((particle_count, 11))

x_list = np.zeros(particle_count)
y_list = np.zeros(particle_count)
u_list = np.zeros(particle_count)
v_list = np.zeros(particle_count)
ro_list = np.full(particle_count, RO_0)
p_list = np.zeros(particle_count)
vol_list = np.full(particle_count, SPACING ** 2)
new_u_list = np.zeros(particle_count)
new_v_list = np.zeros(particle_count)
new_ro_list = np.zeros(particle_count)
particle_type_list = np.zeros(particle_count) #0=fluid, 1=wall

### GENERATING PARTICLE COORDS

index = 0
### FLUID PARTICLE GENERATOR
for i in range(int(FIELD_SIZE / 2)):
	for j in range(FIELD_SIZE):
		x = (WALL_THICKNESS + i) * SPACING
		y = (WALL_THICKNESS + j) * SPACING
		x_list[index] = x
		y_list[index] = y
		particle_type_list[index] = 0
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
			x = i * SPACING
			y = j * SPACING
			x_list[index] = x
			y_list[index] = y
			particle_type_list[index] = 1
			index = index + 1

### GENERATING BOXES
span_x = np.max(x_list)
span_y = np.max(y_list)
boxes_x = int(span_x / BOX_SIZE) + 1
boxes_y = int(span_y / BOX_SIZE) + 1
box_list = []
for i in range(boxes_x):
	box_list.append([])
	for j in range(boxes_y):
		box_list[-1].append([])

### SORTING WALL PARTICLES INTO BOXES, ONCE FOR ALL
wall_boxes = copy.deepcopy(box_list)
for i in range(len(x_list)):
	if particle_type_list[i] == 1:
		box_x = int(x_list[i] / BOX_SIZE)
		box_y = int(y_list[i] / BOX_SIZE)
		wall_boxes[box_x][box_y].append(i)

mymanager = Manager()

mainloop()
