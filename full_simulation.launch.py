import numpy as np
import math
from time import sleep
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import tf2_ros
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import String
from tf2_ros import TransformListener

class PathFollower(Node):

    def __init__(self):
        super().__init__('path_follower')
        
        # Path Storage Parameters
        self.path = []

        # set arbitrarily high to get through first ready check, make sure it matches the if condition in path storage
        self.current_path_index = 1000

        self.goal_pose = []
        self.last_goal = None
        self.goal_reached_msg_sent = False
        self.ready_check = True

        # Transform Parameters to retrieve robot pose in the odom frame
        self.dt = 1.0 / 10.0
        self._tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._to_frame = 'map'
        self._from_frame = 'base_link'

        # Subscriber for messages containing the path (assumes the message is in map frame)
        self.path_sub = self.create_subscription(Path, '/path', self.path_storage, 1)

        # Publisher for goal pose messages (map frame)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # Timer for updating the path point the robot should move to next
        self.timer = self.create_timer(self.dt, self.path_update)

        # Publisher to inform the Path planner to create a path for the next goal marker in the maze
        self.goal_reached_pub = self.create_publisher(String, '/goal_reached', 10)

        self.current_maze_goal_sub = self.create_subscription(PoseStamped, '/current_maze_goal', self.goal_marker_update, 10)

    def path_storage(self,msg):
        if self.current_path_index < int(0.5 * (len(self.path)-1)):
            self.ready_check = False
        elif self.current_path_index > int(0.5 * (len(self.path)-1)) and self.current_path_index != 1000:
            self.ready_check = True
        if self.ready_check is True:
            self.path = msg.poses
            self.current_path_index = 0
        if len(self.path) > 0: 
            goal = self.path[0].pose.position
            self.goal_pose = [goal.x, goal.y]
            self.last_goal = None
        self.goal_reached_msg_sent = False
    
    def goal_marker_update(self, msg):
        self.maze_goal = (msg.pose.position.x, msg.pose.position.y)
        return

    def path_update(self):
        try:
            when = rclpy.time.Time()
            trans = self._tf_buffer.lookup_transform(self._to_frame, self._from_frame, when)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().info('No transform between map and base link')
            return
        
        # Check if goal pose exists:
        if not self.goal_pose or len(self.goal_pose) < 2:
            return
        
        
        # Check how close the robot pose is to the point in the path (in the map frame)
        Robot_pose = [trans.transform.translation.x, trans.transform.translation.y]
        dist = math.sqrt((Robot_pose[0] - self.goal_pose[0])**2 + (Robot_pose[1] - self.goal_pose[1])**2)

        if hasattr(self, "maze_goal") and not self.goal_reached_msg_sent:
            goal_x, goal_y = self.maze_goal
            dist_goal = math.sqrt((Robot_pose[0] - goal_x)**2 + (Robot_pose[1] - goal_y)**2)
            if dist_goal <= 0.3:
                self.get_logger().info('End of Path has been reached')

                goal_reached_msg = String() 
                goal_reached_msg.data = 'Goal reached, Change marker'

                self.goal_reached_pub.publish(goal_reached_msg)
                self.goal_reached_msg_sent = True

        self.get_logger().info(f'Robot at ({Robot_pose[0]:.3f}, {Robot_pose[1]:.3f}), '
                          f'Goal at ({self.goal_pose[0]:.3f}, {self.goal_pose[1]:.3f}), '
                          f'Distance: {dist:.3f}, Index: {self.current_path_index}/{len(self.path)-1}')
        
        if dist <= 0.55:
             if self.current_path_index < len(self.path) - 1:
                self.current_path_index += 1
                self.get_logger().info('Path Index was updated')
             else:
                return

        goal = self.path[self.current_path_index].pose.position
        self.goal_pose = [goal.x, goal.y]
        if self.last_goal is None or abs(self.last_goal[0] - self.goal_pose[0]) > 0.01 or abs(self.last_goal[1] - self.goal_pose[1]) > 0.01:
            goal_msg = PoseStamped()
            goal_msg.header.frame_id = 'map'
            goal_msg.header.stamp = self.get_clock().now().to_msg()
            goal_msg.pose.position.x = float(self.goal_pose[0])
            goal_msg.pose.position.y = float(self.goal_pose[1])
            goal_msg.pose.position.z = 0.0
            goal_msg.pose.orientation.x = 0.0
            goal_msg.pose.orientation.y = 0.0
            goal_msg.pose.orientation.z = 0.0
            goal_msg.pose.orientation.w = 1.0

            self.goal_pub.publish(goal_msg)
            self.last_goal = [self.goal_pose[0], self.goal_pose[1]]

def main(args=None):
    rclpy.init(args=args)
    node = PathFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
