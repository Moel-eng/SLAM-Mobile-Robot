import signal
import sys

import numpy as np
from typing import List

from time import sleep

class MapType:
	UNKNOWN = -1
	UNOCCUPIED = 0
	OCCUPIED = 1


class QuadMap:
	"""
	This class defines the parameters of a mobile agent that avoids collisions
	"""
	def __init__(self, 
				max_depth = 3,
				size = 4.0,
				origin = np.array([0.0,0.0])):

		"""
		Constructor for the QuadMap.

		Input
		  :param max_depth: The maximum depth of the quadtree.
		  :param size: The total height and width of the quadmap (i.e. the size of the root node)
		  :param origin: The centerpoint for the quadmap 
		"""
		self.max_depth = max_depth
		self.size = size
		self.origin = origin
		self.ray_step_size = self.size  / np.power(2, self.max_depth+1)
		self.depth = 0
		self.root = QuadMapNode(None, origin, size)
		self.fig = None

	def point_update(self, point: np.ndarray, state: int):
		"""
		Updates the value of the given point in the QuadMap. Update the value at the maximum depth.

		If the node located at this point already has this state do nothing.

		If the node located at this point is a leaf node at the maximum depth then update is value.

		If the node located at this point is a leaf node not at maximum depth then split it and repeat. 

		After updating the value of the point with its new state the quadmap should collapse any 
		nodes where all the children have the same value. 

		Worth 60 pts

		Input
		  :param point: A 2 element np.ndarray containing the x and y coordinates of the point
		  :param state: An integer indicating the MapType of the point
		"""
		depth = 0
		node = self.root
		while(depth < self.max_depth):
			if node.state == state:
				return
			if len(node.children)==0:
				node.split()
			depth+=1
			id_offset = 0
			delta = abs(point - node.origin)
			if delta[0] > node.size / 2 or delta[1] > node.size / 2 :
				return 
			if point[0] < node.origin[0]:
				id_offset += 2
			if point[1] < node.origin[1]:
				id_offset += 1
			node = node.children[id_offset]
		node.state = state
		node.recursive_combine()

	def ray_update(self, origin: np.ndarray, endpoint: np.ndarray):
		"""
		This function update the map based upon the return of a rangefinding beam.

		This function should update the map to indicate that all the space between the origin
		and the endpoint is unoccupied. Then update the endpoint as occupied.

		Worth 10 pts

		Input
		  :param origin: A 2 element ndarray indicating the location of the robot
		  :param endpoint: A 2 element ndarray indicating the location of the end of the beam
		"""
		vec = endpoint-origin
		vec_len = np.linalg.norm(vec)
		num_inc = int(vec_len / self.ray_step_size)
		for i in range(num_inc):
			point = origin + self.ray_step_size * i * vec / vec_len
			self.point_update(point, MapType.UNOCCUPIED)
		self.point_update(endpoint, MapType.OCCUPIED)

		# Inflate obstacles to avoid robot bumping
		inflation = 0.2
		for x_inflation in [-inflation, 0, inflation]:
			for y_inflation in [-inflation, 0, inflation]:
				if x_inflation == 0 and y_inflation == 0:
					continue
				self.point_update(endpoint + np.array([x_inflation, y_inflation]), MapType.OCCUPIED)
	def get_state(self, point: np.ndarray):
		"""
		Returns the MapType state (-1 for unknown, 0 for unoccupied, 1 for occupied) for a given point

		Worth 10 pts


		Output
		  :return: state: an integer representing the state of that point in space
		"""
		node = self.root
		while len(node.children) > 0:
			id_offset = 0
			if point[0] < node.origin[0]:
				id_offset += 2
			if point[1] < node.origin[1]:
				id_offset += 1
			node = node.children[id_offset]

		return node.state

	def to_occupancygrid(self):
		"""
		Converts the QuadMap into the data element of an occupancy grid. 
		A flattened representation used by ROS. 

		This is a 1-dimensional array with a length equal to the maximum number of nodes in the
		QuadMap. 

		Follow the guidance for the occupancy grid message
		https://github.com/ros2/common_interfaces/blob/master/nav_msgs/msg/OccupancyGrid.msg

		Worth 10 pts

		Output
		  :return: data: a list of integers representing the node values
		"""
		num_leaf = np.power(2, self.max_depth)
		grid = np.zeros((num_leaf,num_leaf))
		self.root.recursive_get_grid(grid)

		grid = np.flipud(grid)

		occupancy_grid = np.zeros_like(grid, dtype=np.int8)
		occupancy_grid[grid == MapType.UNKNOWN] = -1
		occupancy_grid[grid == MapType.UNOCCUPIED] = 0
		occupancy_grid[grid == MapType.OCCUPIED] = 100

		return occupancy_grid.flatten().tolist()
	
class QuadMapNode:
	"""
	A partially implemented Node class for constructing a quadtree. 
	The student is welcome to modify this template implementation by adding
	new class variables or functions, or modifying existing variables and functions.

	The individual functions in QuadMapNode are not graded.

	Each QuadMapNode represents a node in a tree. Where every node other than the root
	has one parent and either 0 children or 4 children. Each node is half the length and width of
	its parent (1/4 the area). 
	"""
	def __init__(self, parent, origin: np.ndarray, size = None, state = MapType.UNKNOWN):
		"""
		Default constructor for the QuadMapNode.

		Input
		  :param parent: The parent node (if this node is root parent=None)
		  :param origin: A 2 element np.ndarray containing the centerpoint of the node
		  :param size: A float describing the height and width of the node

		"""
		if parent is None:
			self.parent = None
			assert size is not None
			self.size = size
		else:
			self.parent = parent
			self.size = parent.size / 2.0 

		self.origin = origin # The origin is the centerpoint of the node

		self.children = [] # A list of all child nodes. Should only contain 0 or 4 children.
		
		self.state = state 


	def split(self):
		"""
		Splits  node into 4 smaller nodes. This function should only be called if the current node is a leaf.

		A useful function to implement for your quadmap
		"""
		self.children.append(QuadMapNode(self, 
			self.origin+self.size/4*np.array([1.0,1.0]), self.size / 2.0, self.state))
		self.children.append(QuadMapNode(self, 
			self.origin+self.size/4*np.array([1.0,-1.0]), self.size / 2.0, self.state))
		self.children.append(QuadMapNode(self, 
			self.origin+self.size/4*np.array([-1.0,1.0]), self.size / 2.0, self.state))
		self.children.append(QuadMapNode(self, 
			self.origin+self.size/4*np.array([-1.0,-1.0]), self.size / 2.0, self.state))

	def combine(self):
		"""
		Checks to see if all the children of the node have the same type, if they do then delete them and assign
		the current node that type so that it becomes a leaf node.

		A useful function to implement for your quadmap
		"""
		if len(self.children) > 0:
			child_state = self.children[0].state
			for child in self.children:
				if child.state != child_state:
					return False
			self.children = []
			self.state = child_state
			return True
		return True

	def recursive_combine(self):
		if self.combine():
			self.parent.recursive_combine()

	def recursive_get_grid(self, ngrid: np.ndarray):
		"""
		Recursively slices the grid into smaller and smaller quadrants.
		Then sets the value of all cells in grid to the value of the leaf
		node. 

		This function takes advantage of the fact that python passes
		by reference to modify the larger array.

		Input
		  :param node: An instance of the QuadMapNode class that has a value and children
		  :param ngrid: An instance of an np.ndarray containing the slice of gridcells
		  				representing that node

		"""
		l = len(ngrid)
		l_half = int(l / 2)
		if self.children is None or len(self.children) == 0:
			ngrid[:] = self.state
		else:
			self.children[0].recursive_get_grid(ngrid[:l_half,l_half:])
			self.children[1].recursive_get_grid(ngrid[l_half:,l_half:])
			self.children[2].recursive_get_grid(ngrid[:l_half,:l_half])
			self.children[3].recursive_get_grid(ngrid[l_half:,:l_half])
