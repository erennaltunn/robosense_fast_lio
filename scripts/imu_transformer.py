#!/usr/bin/env python
import rospy
import numpy as np
import tf_conversions
from sensor_msgs.msg import Imu
import tf2_ros
from geometry_msgs.msg import TransformStamped

def quat_mul(a, b):
    return tf_conversions.transformations.quaternion_multiply(a, b)

def quat_from_rpy(roll, pitch, yaw):
    return tf_conversions.transformations.quaternion_from_euler(roll, pitch, yaw)

def rotmat_from_quat(q):
    return tf_conversions.transformations.quaternion_matrix(q)[:3, :3]

class ImuTransformer:
    def __init__(self):
        # we now default to a NEW frame name
        self.target_frame = rospy.get_param("~target_frame", "r5000219a8/front_imu_link")
        # parent is your robot base (can be namespaced)
        self.parent_frame = rospy.get_param("~parent_frame", "r5000219a8/base_link")

        self.rpy_deg = rospy.get_param("~rpy_deg", [90.0, 0.0, 0.0])
        self.accel_scale = rospy.get_param("~accel_scale", 9.80665)
        self.gyro_scale  = rospy.get_param("~gyro_scale", 1.0)

        rr, rp, ry = np.deg2rad(self.rpy_deg)
        self.q_ST = quat_from_rpy(rr, rp, ry)   # source -> target rotation
        self.R_ST = rotmat_from_quat(self.q_ST)

        # --- static TF: base_link -> front_imu_link ---
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        static_tf = TransformStamped()
        static_tf.header.stamp = rospy.Time.now()
        static_tf.header.frame_id = self.parent_frame      # e.g. r5000219a8/base_link
        static_tf.child_frame_id  = self.target_frame      # e.g. r5000219a8/front_imu_link

        # zero translation
        static_tf.transform.translation.x = 1.316
        static_tf.transform.translation.y = 0.0
        static_tf.transform.translation.z = 0.7905

        # identity rotation
        static_tf.transform.rotation.x = 0.0
        static_tf.transform.rotation.y = 0.0
        static_tf.transform.rotation.z = 0.0
        static_tf.transform.rotation.w = 1.0

        self.static_broadcaster.sendTransform(static_tf)
        rospy.loginfo("Published static TF %s -> %s",
                      self.parent_frame, self.target_frame)

        # IMU I/O
        self.sub = rospy.Subscriber("imu_in", Imu, self.cb, queue_size=200)
        self.pub = rospy.Publisher("imu_out", Imu, queue_size=200)

    def cb(self, msg):
        out = Imu()
        out.header = msg.header
        # IMU will now live on front_imu_link
        out.header.frame_id = self.target_frame

        # Orientation
        qS = np.array([msg.orientation.x,
                       msg.orientation.y,
                       msg.orientation.z,
                       msg.orientation.w])
        if np.allclose(qS, 0.0):
            out.orientation.x = out.orientation.y = out.orientation.z = out.orientation.w = 0.0
            cov = list(msg.orientation_covariance)
            if len(cov) == 9:
                cov[0] = -1.0
            out.orientation_covariance = cov
        else:
            qT = quat_mul(self.q_ST, qS)
            out.orientation.x, out.orientation.y, out.orientation.z, out.orientation.w = qT
            out.orientation_covariance = msg.orientation_covariance

        # Angular velocity
        wS = np.array([msg.angular_velocity.x,
                       msg.angular_velocity.y,
                       msg.angular_velocity.z]) * self.gyro_scale
        wT = self.R_ST.dot(wS)
        out.angular_velocity.x, out.angular_velocity.y, out.angular_velocity.z = wT
        out.angular_velocity_covariance = msg.angular_velocity_covariance

        # Linear acceleration
        aS = np.array([msg.linear_acceleration.x,
                       msg.linear_acceleration.y,
                       msg.linear_acceleration.z]) * self.accel_scale
        aT = self.R_ST.dot(aS)
        out.linear_acceleration.x, out.linear_acceleration.y, out.linear_acceleration.z = aT
        out.linear_acceleration_covariance = msg.linear_acceleration_covariance

        self.pub.publish(out)

if __name__ == "__main__":
    rospy.init_node("imu_transformer")
    ImuTransformer()
    rospy.spin()
