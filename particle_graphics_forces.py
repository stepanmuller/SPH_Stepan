from tkinter import *
import copy

class Graphics:
	def __init__(self, window):
		self.window = window
		self.screen_width = 800 
		self.screen_height = 800
		self.canvas = Canvas(self.window, width = self.screen_width, height = self.screen_height, bg="white")
		self.canvas.pack()

	def update(self, x_list, y_list, p_list, fx, fy):
		self.x_list = copy.deepcopy(x_list)
		self.y_list = copy.deepcopy(y_list)
		self.p_list = copy.deepcopy(p_list)
		self.fx = fx
		self.fy = fy
		self.get_graphics_coords(self.x_list, self.y_list)
		self.get_graphics_pressure(self.p_list)
		self.draw()
	
	def get_graphics_coords(self, x_l, y_l):
		x_list = []
		y_list = []
		for i in range(len(x_l)):
			x_list.append(x_l[i])
			y_list.append(-y_l[i])
		min_x = min(x_list)
		max_x = max(x_list)
		mid_x = (max_x + min_x) / 2
		min_y = min(y_list)
		max_y = max(y_list)
		mid_y = (max_y + min_y) / 2
		x_span = max_x - min_x
		y_span = max_y - min_y
		x_mult = (self.screen_width - 100) / x_span
		y_mult = (self.screen_height - 100) / y_span
		mult = min([x_mult, y_mult])
		shift_x = (self.screen_width / 2) - mid_x * mult
		shift_y = (self.screen_height / 2) - mid_y * mult
		for i in range(len(self.x_list)):
			self.x_list[i] = self.x_list[i] * mult + shift_x
			self.y_list[i] = -self.y_list[i] * mult + shift_y
	
	def get_graphics_pressure(self, p_list):
		self.graphics_p_list = []
		for p in p_list:
			self.graphics_p_list.append(p)
		p_min = min(self.graphics_p_list)
		p_max = max(self.graphics_p_list)
		span = p_max - p_min
		for index in range(len(self.graphics_p_list)):
			pressure = self.graphics_p_list[index]
			if span == 0:
				span = 1
			self.graphics_p_list[index] = (pressure - p_min) / span
			
	def draw(self):
		self.canvas.delete("all")
		for i in range(len(self.x_list)):
			x = self.x_list[i]
			y = self.y_list[i]
			p = self.graphics_p_list[i]
			rgb = self.get_RGB(p)
			self.canvas.create_rectangle(x-3, y-3, x+3, y+3, fill = rgb)
			fx_length = self.fx[i] * 10
			fy_length = self.fy[i] * 10
			self.canvas.create_line(x, y, x + fx_length, y - fy_length, fill = "black")
			self.canvas.create_text(x, y-8, text = i, font = ("ISOCPEUR", "10"), fill = rgb)
	
	def get_RGB(self, value):
		R = int(255 * (value**2))
		G = int(255 * 2 * value * (1 - value))
		B = int(255 * (1 - value)**2)
		rgb = (R, G, B)
		return "#%02x%02x%02x" % rgb

