import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
import numpy as np
from geometry_msgs.msg import Twist, Quaternion, PoseStamped, PointStamped
import tf2_ros
from tf2_ros import TransformListener
from tf2_geometry_msgs import do_transform_point
from time import sleep
from rclpy.duration import Duration
import math
from gazebo_controller.mapping import QuadMap

class MapPublisher(Node):
    def __init__(self):
        super().__init__('map_publisher') 

        # Setup to retrieve robot pose in world frame
        self.dt = 1.0 / 60.0
        self._tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._to_frame = 'map'
        self._from_frame = 'base_link'
        self._from_lidar = 'vehicle_blue/lidar/lidar_sensor'

        # LaserScan Subscriber
        self.scan_subscriber = self.create_subscription(LaserScan, '/lidar', self.listener_callback, 10)

        # Occupancy grid msg Publisher
        self.map_publisher = self.create_publisher(OccupancyGrid,'/map', 10)

        # Create the Quadmap
        self.map = QuadMap(max_depth = 7,
				size = 20.0,
				origin = np.array([0.0,0.0]))
        self.output_resolution = float(self.map.size) / (2 ** self.map.max_depth)
        self.output_grid_size = int(2 ** self.map.max_depth)

    def listener_callback(self, msg):
        try:
            when = rclpy.time.Time()
            trans_robot = self._tf_buffer.lookup_transform(self._to_frame, self._from_frame, when, timeout=Duration(seconds=5))
            trans_laser = self._tf_buffer.lookup_transform(self._to_frame,self._from_lidar, when, timeout=Duration(seconds=5))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            return
        
        # Initialize ray angle starting from minimum
        angle = msg.angle_min
        Lidar_Position = np.array([trans_laser.transform.translation.x, trans_laser.transform.translation.y])
        for i,r in enumerate(msg.ranges):
            if r < msg.range_min or r > msg.range_max or np.isnan(r) or np.isinf(r) or r <= 0.0:
                angle += msg.angle_increment
                continue

            # max distance at which a hit is regarded as relevant
            hit_mark_treshold = 3.0 
            if r < hit_mark_treshold:
                lidar_endpoint = PointStamped()
                lidar_endpoint.header.frame_id = msg.header.frame_id
                lidar_endpoint.header.stamp = msg.header.stamp
                lidar_endpoint.point.x = float(math.cos(angle) * r)
                lidar_endpoint.point.y = float(math.sin(angle) * r)
                lidar_endpoint.point.z = 0.0

                map_endpoint = do_transform_point(lidar_endpoint, trans_laser)
                endpoint = np.array([map_endpoint.point.x, map_endpoint.point.y])
                self.map.ray_update(Lidar_Position, endpoint)

            angle += msg.angle_increment
        # Construct the Occupancy Grid msg
        map_msg = OccupancyGrid()
        
        # Header
        map_msg.header = Header()
        map_msg.header.stamp = self.get_clock().now().to_msg()

        # Might need to be edited
        map_msg.header.frame_id = 'map'
        
        # Map metadata
        map_msg.info.resolution = self.output_resolution
        map_msg.info.width = self.output_grid_size
        map_msg.info.height = self.output_grid_size
        map_msg.info.origin.position.x = float(-self.map.size / 2)
        map_msg.info.origin.position.y = float(-self.map.size / 2)
        map_msg.info.origin.position.z = float(0.0)
        map_msg.info.origin.orientation.w = float(1.0)
        map_msg.data = self.map.to_occupancygrid()

        self.map_publisher.publish(map_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MapPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




