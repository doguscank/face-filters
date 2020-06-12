import cv2
import numpy as np
import random

def random_direction():
	directions = ['v', 'h', 'vr', 'hr']
	return directions[random.randint(0, 3)]

class Creator():
	def __init__(self, debug = False):
		self.debug = debug

	#Main square edge length, number of squares to be drawn, gradient start, gradient end, direction of gradient,
	#random gradient directions, max square edge multiplier, put squares from bigger to smaller into the frame
	def create_squares_background(self, max_l, num_squares, first_color, second_color, direction = 'h', random_directions = False,
	percentage = 0.9, sort_squares = True):
		square_creator = Square(max_l, first_color, second_color, direction, random_directions)
		main_square = square_creator.create_square(max_l)

		percentage = min(0.9, percentage)

		squares = list()

		for i in range(num_squares - 1):
			l = int(max_l * random.uniform(0, percentage))
			new_square = square_creator.create_square(l)

			rx = int((max_l - l) * random.uniform(0, percentage)) #Random X point of center of the new square
			ry = int((max_l - l) * random.uniform(0, percentage)) #Random Y point of center of the new square
			
			if self.debug:		
				print(l, rx, ry, rx + l, ry + l)

			squares.append([l, rx, ry, new_square])

		if sort_squares:
			squares = sorted(squares, key = lambda square: square[0], reverse = True)

		for s in squares:
			main_square[s[1]: s[1] + s[0], s[2]: s[2] + s[0]] = s[3]

		return main_square

class Square():
	def __init__(self, l, start_color = (255, 255, 255), end_color = None, direction = 'h', random_directions = False, debug = False):
		self.l = l
		self.start_color = start_color
		self.debug = debug
		self.random_directions = random_directions

		if end_color:
			self.end_color = end_color
		else:
			self.end_color = start_color

		if self.random_directions:
			self.direction = random_direction()
		else:
			self.direction = direction

	#Create square based on given parameters
	def create_square(self, l = None):
		#If a new length parameter is given, update current parameter
		if l:
			self.update_length(l)

		#If random directions are enabled, select a random direction
		if self.random_directions:
			new_direction = random_direction()

			if 'r' in new_direction:
				self.update_colors(self.end_color, self.start_color)

			self.direction = new_direction[0]

		#If no color gradient is required, return flat colored square
		if self.start_color == self.end_color:
			b, g, r = self.start_color
			self.square = np.full((self.l, self.l, 3), (b / 255.0, g / 255.0, r / 255.0))

			return self.square

		#Create white colored main square
		self.square = np.full((self.l, self.l, 3), (1.0, 1.0, 1.0))

		#Get start and end values of BGR values
		[SB, SG, SR] = [i / float(255.0) for i in self.start_color]
		[EB, EG, ER] = [i / float(255.0) for i in self.end_color]

		if self.debug:
			print("{} {} {} {} {} {}".format(SB, EB, SG, EG, SR, ER))
		
		#Create gradient based on current direction
		if self.direction.lower() == 'h':
			for i in range(self.l):
				self.square[:, i] = (SB + (EB - SB) * i / float(self.l), SG + (EG - SG) * i  / float(self.l), SR + (ER - SR) * i  / float(self.l))

				if self.debug: 
					print((SB + (EB - SB) * i / float(self.l), SG + (EG - SG) * i  / float(self.l), SR + (ER - SR) * i  / float(self.l)))

		elif self.direction.lower() == 'v':
			for i in range(self.l):
				self.square[i] = (SB + (EB - SB) * i / float(self.l), SG + (EG - SG) * i  / float(self.l), SR + (ER - SR) * i  / float(self.l))

				if self.debug: 
					print((SB + (EB - SB) * i / float(self.l), SG + (EG - SG) * i  / float(self.l), SR + (ER - SR) * i  / float(self.l)))

		return self.square

	#Update color parameters of square creator
	def update_colors(self, start_color, end_color = None):
		self.start_color = start_color

		if end_color:
			self.end_color = end_color
		else:
			self.end_color = start_color

	def update_length(self, l):
		if l > 0:
			self.l = l
		else:
			self.l = 1
			print("Invalid length value given! New value is 1px.")

		return self.l

	#Create n points for triangles
	def create_rnd_points(self, n, corner_threshold = 10):
		points = list()

		for i in range(n):
			#Get random X and Y point
			rx = random.randint(corner_threshold - 1, self.l - corner_threshold - 1)
			ry = random.randint(corner_threshold - 1, self.l - corner_threshold - 1)

			points.append((rx, ry))

		return points

	def triangulate(self, n = 10, gradient = False, start_color = (0, 0, 0), end_color = (255, 255, 255)):
		if gradient:
			self.square = np.full((self.l, self.l, 3), (0.0, 0.0, 0.0))

		points = self.create_rnd_points(n)
		points.extend([(1, 1), (1, self.l - 2), (self.l - 2, self.l - 2), (self.l - 2, 1)])
		
		subdiv = cv2.Subdiv2D((0, 0, self.l - 1, self.l - 1))

		for p in points:
			subdiv.insert(p)

		triangles = subdiv.getTriangleList()
		triangle_list = list()
		
		for t in triangles:
			pts = np.array([(t[0], t[1]), (t[2], t[3]), (t[4], t[5])], np.int32)
			pts = pts.reshape((-1, 1, 2))
			#If gradient is required, 
			if gradient:
				new_t = Triangle(pts)
				triangle_list.append(new_t)
				#Get masked gradient triangle
				t_grad = new_t.create_gradient_triangle(self.l, start_color = self.get_random_color(start_color), end_color = self.get_random_color(end_color), random_directions = self.random_directions)
				self.square = cv2.bitwise_or(self.square, t_grad)
			else:
				rb, rg, rr = random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)
				cv2.fillConvexPoly(self.square, pts, (rb, rg, rr))

		return self.square

	def get_random_color(self, start = (0, 0, 0), end = (255, 255, 255)):
		bs, gs, rs = start
		be, ge, re = end

		rb, rg, rr = random.randint(bs, be), random.randint(gs, ge), random.randint(rs, re)

		return (rb, rg, rr)

class Triangle():
	def __init__(self, pts):
		self.pts = pts

	def create_mask(self, l, w = None):
		if not w:
			w = l
		mask_base = np.full((l, w), 0.0, dtype = np.uint8)
		cv2.fillConvexPoly(mask_base, self.pts, 1.0)

		return mask_base

	def create_gradient_triangle(self, l = None, start_color = (255, 255, 255), end_color = None, direction = 'h', random_directions = False, debug = False):
		rect = cv2.boundingRect(self.pts)
		x1, y1, x2, y2 = rect

		if not l:
			l = max(abs(y2 - y1), abs(x2 - x1))

		mask = self.create_mask(l)

		if random_directions:
			new_direction = random_direction()

			if 'r' in new_direction:
				start_color, end_color = end_color, start_color

			direction = new_direction[0]

		sc = Square(l, start_color = start_color, end_color = end_color, direction = direction, random_directions = random_directions, debug = debug)
		new_square = sc.create_square()
		if debug:
			cv2.imshow('mask', mask)
			cv2.waitKey(0)

			cv2.imshow('new_square', new_square)
			cv2.waitKey(0)
		
		result = cv2.bitwise_and(new_square, new_square, mask = mask)
		
		if debug:
			cv2.imshow('result', result)
			cv2.waitKey(0)

		return result