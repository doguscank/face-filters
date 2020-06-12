import cv2
import numpy as np
import random
import dlib
from collections import defaultdict
import shape_lib as sl

class FaceDetector:
	def __init__(self):
		self.init_detector()

		self.t_creation_pts = None

	#Initializes face detector
	def init_detector(self):
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor('./dat/shape_predictor_68_face_landmarks.dat')

	#Detect face landmark points on given image
	#img: Face detector will be work on this image
	def detect_face(self, img):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		faces = self.detector(gray)

		landmark_dict = defaultdict(int)

		for j, face in enumerate(faces):
			landmarks = self.predictor(gray, face)
			landmark_points = list()

			for i in range(68):
				x = landmarks.part(i).x
				y = landmarks.part(i).y

				landmark_points.append((x, y))

			landmark_dict[j] = landmark_points

		return landmark_dict

	#Draw green bounding box of face on given face image
	#img: Main image to be used as background
	#landmark_dict: Includes points obtained using face landmark detection
	def draw_face_rect(self, img, landmark_dict):
		img_copy = img.copy() #Copy image to not affect the original one

		for landmark in landmark_dict:
			landmark_points = landmark_dict[landmark]

			landmark_points = np.int32(landmark_points)
			#Convex hull is the polygon that covers all the points given
			convex_hull = cv2.convexHull(landmark_points)
			#Bounding rect is the rectangle that covers all the points given
			(x, y, w, h) = cv2.boundingRect(convex_hull)

			cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)

		return img_copy

	#Draw green landmark points on given face image
	#img: Main image to be used as background
	#landmark_dict: Includes points obtained using face landmark detection
	def draw_face_pts(self, img, landmark_dict):
		img_copy = img.copy() #Copy image to not affect the original one

		for landmark in landmark_dict:
			landmark_points = landmark_dict[landmark]

			for pt in landmark_points:
				cv2.circle(img_copy, pt, 2, (0, 255, 0), -1)

		return img_copy

	#Runs Delaunay Triangulation based on given points
	#img: Main image to be used as background
	#landmark_dict: Includes points obtained using face landmark detection
	#colors: If given, use this colors, else, get random colors
	#remove_bg: If True, use white background
	#generate_bg: If True, use background image given or generate a random background
	#bg_given: If given and "generate_bg" parameter is True, use it as background of image
	def triangulation(self, img, landmark_dict, colors = None, remove_bg = False,
		generate_bg = True, bg_given = None):
		if remove_bg:
			img_copy = np.full(img.shape, 255, dtype = np.uint8)
		elif generate_bg:
			if bg_given is not None:
				img_copy = bg_given
			else:
				img_copy = self.generate_square_background(img)
		else:
			img_copy = img.copy() #Copy image to not affect the original one

		bg = None

		for landmark in landmark_dict:
			landmark_points = landmark_dict[landmark]
			#Landmark points are covnerted to int32 in order to use construct convex hull
			hull_pts = np.int32(landmark_points)

			convex_hull = cv2.convexHull(hull_pts)
			hull_rect = cv2.boundingRect(convex_hull)

			#Create subdiv with boundary edges of bounding rect of the convex hull
			subdiv = cv2.Subdiv2D(hull_rect)
			subdiv.insert(landmark_points)

			#Get Delaunay triangles from subdiv
			triangles = subdiv.getTriangleList()

			if not self.t_creation_pts:
				self.t_creation_pts = defaultdict(int)

				for i, t in enumerate(triangles):
					corner_points = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
					c_index_1 = np.where((corner_points[0] == hull_pts).all(axis = 1))[0][0]
					c_index_2 = np.where((corner_points[1] == hull_pts).all(axis = 1))[0][0]
					c_index_3 = np.where((corner_points[2] == hull_pts).all(axis = 1))[0][0]
					
					self.t_creation_pts[i] = (c_index_1, c_index_2, c_index_3)

			for i in self.t_creation_pts:
				if colors:
					(rb, rg, rr) = colors[i]
				else:
					(rb, rg, rr) = self.pick_random_colors(1)[0]

				p_1 = landmark_points[self.t_creation_pts[i][0]]
				p_2 = landmark_points[self.t_creation_pts[i][1]]
				p_3 = landmark_points[self.t_creation_pts[i][2]]

				t_pts = np.array([p_1, p_2, p_3], np.int32)
				t_pts = t_pts.reshape((-1, 1, 2))
				cv2.fillConvexPoly(img_copy, t_pts, (rb, rg, rr))
		
		return img_copy

	#Creates Voronoi Diagrams based on given points
	#img: Main image to be used as background
	#landmark_dict: Includes points obtained using face landmark detection
	#colors: If given, use this colors, else, get random colors
	#remove_bg: If True, use white background (currently obsolete)
	def voronoi(self, img, landmark_dict, colors = None, remove_bg = False):
		if remove_bg:
			img_copy = np.full(img.shape, 255, dtype = np.uint8)
		else:
			img_copy = img.copy() #Copy image to not affect the original one

		bg = None

		for landmark in landmark_dict:
			landmark_points = landmark_dict[landmark]
			hull_pts = np.int32(landmark_points)

			size = img.shape
			rect = (0, 0, size[1], size[0])

			#Create subdiv with the whole image
			subdiv = cv2.Subdiv2D(rect)
			subdiv.insert(landmark_points)

			(facets, centers) = subdiv.getVoronoiFacetList([])

			for i in range(len(facets)):
				if colors:
					(rb, rg, rr) = colors[i]
				else:
					(rb, rg, rr) = self.pick_random_colors(1)[0]

				ith_facet = list()

				for f in facets[i]:
					ith_facet.append(f)

				ith_facet = np.array(ith_facet, np.int)

				cv2.fillConvexPoly(img_copy, ith_facet, (rb, rg, rr))
		
		return img_copy
	
	#Uses Delaunay Triangulation method to create triangles on given face
	#cam_id: ID of the camera wanted to be used
	#resizeable: Resizeable window
	#num_color_sets: Number of colors sets to be switch
	#change_per_k_frame: Frame number to change the color set
	#generate_bg: If False, use flat white background, else create a random background
	#generate_bg_once: If True, create background once and use it in every frame
	#Note: Creating backgrounds in every iteration is not recommended!
	#use_white_bg: Uses white background without taking care of other parameters
	#triangulate_bg: If true, created background will be triangulated
	def triangulate_cam_face(self, cam_id = 0, resizeable = True, num_color_sets = 2,
		change_per_k_frame = 20, generate_bg = True, generate_bg_once = True, use_white_bg = False,
		triangulate_bg = True):
		cap = cv2.VideoCapture(cam_id)

		i = 0 #Color change counter
		n = num_color_sets #Number of color sets

		j = 0 #Frame counter
		k = change_per_k_frame #Frame limit

		generated_bg = False
		bg_to_use = None
		bg_to_use_c = None

		#If "use_white_bg" is True, do not take care of other background parameters
		if use_white_bg:
			generate_bg_once = True
			generate_bg = False

		colors = [self.pick_random_colors(130, is_int = not generate_bg) for k in range(n)]

		if resizeable:
			cv2.namedWindow('result', cv2.WINDOW_NORMAL)

		while True:
			ret, frame = cap.read()

			if generate_bg:
				if generate_bg_once:
					if not generated_bg:
						bg_to_use = self.generate_square_background(frame, triangulate = triangulate_bg)
						generated_bg = True
				else:
					bg_to_use = self.generate_square_background(frame, triangulate = triangulate_bg)

			if bg_to_use is not None:
				bg_to_use_c = bg_to_use.copy()

			face_rect = self.triangulation(frame, self.detect_face(frame), colors = colors[i],
				remove_bg = use_white_bg, generate_bg = generate_bg, bg_given = bg_to_use_c)
			cv2.imshow('result', face_rect)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break			

			j += 1

			if j % k == 0:
				i = (i + 1) % n
				j = 0

		#Release camera capture and destroy all remaining windows
		cap.release()
		cv2.destroyAllWindows()

	#Creates Voronoi Diagrams on given face
	#cam_id: ID of the camera wanted to be used
	#resizeable: Resizeable window
	#num_color_sets: Number of colors sets to be switch
	#change_per_k_frame: Frame number to change the color set
	def voronoi_cam_face(self, cam_id = 0, resizeable = True, num_color_sets = 2, change_per_k_frame = 20):
		cap = cv2.VideoCapture(cam_id)

		i = 0 #Color change counter
		n = num_color_sets #Number of color sets

		j = 0 #Frame counter
		k = change_per_k_frame #Frame limit

		colors = [self.pick_random_colors(130) for k in range(n)]

		if resizeable:
			cv2.namedWindow('result', cv2.WINDOW_NORMAL)

		while True:
			ret, frame = cap.read()

			face_rect = self.voronoi(frame, self.detect_face(frame), colors = colors[i], remove_bg = True)
			cv2.imshow('result', face_rect)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

			j += 1

			if j % k == 0:
				i = (i + 1) % n
				j = 0

		#Release camera capture and destroy all remaining windows
		cap.release()
		cv2.destroyAllWindows()

	#Picks n random colors
	#n: Number of color sets to be randomly picked
	#is_int: If True, generate colors in range of [0, 255],
	#else, [0.0, 1.0]
	#Note: Created colors are using BGR color space
	def pick_random_colors(self, n, is_int = True):
		colors = list()

		for i in range(n):
			if is_int:
				rb, rg, rr = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
				colors.append((rb, rg, rr))
			else:
				rb, rg, rr = random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)
				colors.append((rb, rg, rr))

		return colors

	#Generates square background to be used as background of triangulation filter
	#ref_img: Reference image to get shape
	#triangulate: If True, triangulate generated background
	def generate_square_background(self, ref_img, triangulate = True):
		l = max(ref_img.shape) #Get max edge length

		sc = sl.Square(800, (0, 0, 150), (0, 0, 255), random_directions = False, debug = False)
		square = sc.create_square()

		if triangulate:
			bg = sc.triangulate(25, start_color = (0, 0, 150), end_color = (0, 0, 255), gradient = True)
			return bg

		return square

if __name__ == '__main__':
	fd = FaceDetector()

	fd.triangulate_cam_face(0, generate_bg = True, generate_bg_once = True, use_white_bg = True, triangulate_bg = False)
	#fd.voronoi_cam_face(0)
