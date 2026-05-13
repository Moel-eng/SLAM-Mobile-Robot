import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, OccupancyGrid
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import Point, PoseStamped, PointStamped
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np
import tf2_ros
from tf2_ros import TransformListener
from time import sleep
from rclpy.duration import Duration
from tf2_geometry_msgs import do_transform_point
from gazebo_controller.a_star_search import a_star_grid

class PathPublisher(Node):
    def __init__(self):
        super().__init__('path_publisher')

        self.map = None
        self.grid = None
        self.dt = 1.0 / 60.0
        self._tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._to_frame = 'map'
        self._from_frame = 'base_link'
        self.goal_marker_index = 0
        self.goal_cells = []
        self.path_generated = False
        self.maze_goals = None

        # Subscriber for Map in form of Occupancy Grid message
        self.map_sub = self.create_subscription(OccupancyGrid,'/map', self.convert_to_grid, 10)

        # Subscriber for the goal sphere poses (in map frame)
        self.goal_sphere_sub = self.create_subscription(MarkerArray, '/goal_points', self.convert_to_cell, 10)

        # Publisher for the path in form of Path msg
        self.path_publisher = self.create_publisher(Path, '/path', 10)

        self.current_maze_goal_pub = self.create_publisher(PoseStamped, '/current_maze_goal', 10)

        # Subscriber that will update the goal marker upon reaching current goal
        self.goal_update_sub = self.create_subscription(String, '/goal_reached', self.goal_update, 10)

        # Timer to update the path
        self.path_timer = self.create_timer(1, self.path_publish)
    
    def goal_update(self, msg):
        if self.goal_marker_index >= len(self.goal_cells):
            return 
        self.goal_marker_index += 1
        self.path_generated = False

    def convert_to_grid(self, msg):
        self.map = msg
        width = self.map.info.width
        height = self.map.info.height
        data = self.map.data

        size_expectation = width * height
        if len(data) != size_expectation:
            return
        
        grid = np.zeros((height,width), dtype=int)
        for i in range(height):
            for j in range(width):
                idx = i * width + j
                cell = data[idx]  
                
                if cell == 100:
                    grid[i][j] = 1
                elif cell == -1:
                    grid[i][j] = 0 
                else:
                    grid[i][j] = 0
        
        self.grid = grid
    
    def convert_to_cell(self,msg):
        
        if self.map is None:
            return
        
        # Convert goal markers from the map frame to a grid cell
        resolution = self.map.info.resolution
        origin = self.map.info.origin
        maze_goal_cells = []

        sorted_markers = sorted(msg.markers, key=lambda m: m.id, reverse=True)
        self.maze_goals = sorted_markers

        for m in sorted_markers:
            goal_x = m.pose.position.x
            goal_y = m.pose.position.y
            goal_cell_row = int(np.floor((goal_y - origin.position.y) / resolution))
            goal_cell_col = int(np.floor((goal_x - origin.position.x) / resolution))
            goal_cell = (goal_cell_row, goal_cell_col)
            maze_goal_cells.append(goal_cell)

        self.goal_cells = maze_goal_cells
        
    def path_publish(self):

        if self.map is None:
            return
        if self.goal_cells == []:
            return
        if self.grid is None:
            return
        
        if self.goal_marker_index <= (len(self.maze_goals) - 1):
            maze_goal_x = self.maze_goals[self.goal_marker_index].pose.position.x
            maze_goal_y = self.maze_goals[self.goal_marker_index].pose.position.y

            current_maze_goal_msg = PoseStamped()
            current_maze_goal_msg.header.stamp = self.get_clock().now().to_msg()
            current_maze_goal_msg.header.frame_id = 'map'

            current_maze_goal_msg.pose.position.x = maze_goal_x
            current_maze_goal_msg.pose.position.y = maze_goal_y
            current_maze_goal_msg.pose.position.z = 0.0
            current_maze_goal_msg.pose.orientation.w = 1.0
            self.current_maze_goal_pub.publish(current_maze_goal_msg)
        
        try:
            when = rclpy.time.Time()
            trans = self._tf_buffer.lookup_transform(self._to_frame, self._from_frame, when)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            return
        
        # Robot pose in the map frame
        Robot_x = trans.transform.translation.x
        Robot_y = trans.transform.translation.y

        # Convert to a cell in the grid
        Robot_row = int(np.floor((Robot_y - self.map.info.origin.position.y) / self.map.info.resolution))
        Robot_col = int(np.floor((Robot_x - self.map.info.origin.position.x) / self.map.info.resolution))

        # Check the bounds of the cell
        if Robot_row < 0  or Robot_row >= self.map.info.height or Robot_col < 0 or Robot_col >= self.map.info.width:
            Robot_row = max(0, min(Robot_row, self.map.info.height - 1))
            Robot_col = max(0, min(Robot_col, self.map.info.width - 1))

        # Save as tuple
        Robot_cell = (Robot_row, Robot_col)

        # Set the goal cell as one of the maze markers
        goal_cell = self.goal_cells[self.goal_marker_index]

        # Perform a star grid search using the grid, robot cell as the start, maze marker cell as the end
        path = a_star_grid(self.grid, Robot_cell, goal_cell)

        if not path:
            self.get_logger().warn('No path was made')
            return
        
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        for row, col in path:
            pose = PoseStamped()
            pose.header.stamp = path_msg.header.stamp
            pose.header.frame_id = 'map'

            # Convert grid cells back to poses in map frame
            pose.pose.position.x = col * self.map.info.resolution + self.map.info.origin.position.x + (self.map.info.resolution / 2)
            pose.pose.position.y = row * self.map.info.resolution + self.map.info.origin.position.y + (self.map.info.resolution / 2)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0

            path_msg.poses.append(pose)

        self.path_publisher.publish(path_msg)
        self.get_logger().info("PATH PUBLISHED")
        self.path_generated = True

def main(args=None):
    rclpy.init(args=args)
    node = PathPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
        






    