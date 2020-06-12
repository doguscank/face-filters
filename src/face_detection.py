import cv2
import numpy as np
import random
import dlib
from collections import defaultdict

class FaceDetector:
	def __init__(self):
		self.init_detector()

		self.t_creation_pts = None

	def init_detector(self):
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor('./dat/shape_predictor_68_face_landmarks.dat')

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

	def draw_face_rect(self, img, landmark_dict):
		img_copy = img.copy()

		for landmark in landmark_dict:
			landmark_points = landmark_dict[landmark]

			landmark_points = np.int32(landmark_points)
			convex_hull = cv2.convexHull(landmark_points)

			(x, y, w, h) = cv2.boundingRect(convex_hull)

			cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)

		return img_copy

	def draw_face_pts(self, img, landmark_dict):
		img_copy = img.copy()

		for landmark in landmark_dict:
			landmark_points = landmark_dict[landmark]

			for pt in landmark_points:
				cv2.circle(img_copy, pt, 2, (0, 255, 0), -1)

		return img_copy

	def triangulation(self, img, landmark_dict, colors = None, remove_bg = False):
		if remove_bg:
			img_copy = np.full(img.shape, 255, dtype = np.uint8)
		else:
			img_copy = img.copy()

		bg = None

		for landmark in landmark_dict:
			landmark_points = landmark_dict[landmark]
			hull_pts = np.int32(landmark_points)

			convex_hull = cv2.convexHull(hull_pts)
			hull_rect = cv2.boundingRect(convex_hull)

			subdiv = cv2.Subdiv2D(hull_rect)
			subdiv.insert(landmark_points)

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
		
	def triangulate_cam_face(self, cam_id = 0):
		cap = cv2.VideoCapture(cam_id)
		i = 0 #Color change counter
		n = 2 #Number of color sets

		j = 0 #Frame counter
		k = 10 #Frame limit

		colors = [self.pick_random_colors(130) for k in range(n)]

		while True:
			ret, frame = cap.read()

			face_rect = self.triangulation(frame, self.detect_face(frame), colors = colors[i], remove_bg = True)
			cv2.imshow('result', face_rect)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

			j += 1

			if j % k == 0:
				i = (i + 1) % n
				j = 0

	def pick_random_colors(self, n):
		colors = list()

		for i in range(n):
			rb, rg, rr = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
			colors.append((rb, rg, rr))

		return colors

if __name__ == '__main__':
	fd = FaceDetector()

	fd.triangulate_cam_face(0)