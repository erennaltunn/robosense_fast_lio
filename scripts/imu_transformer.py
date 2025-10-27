#!/usr/bin/env python
import rospy
import numpy as np
import tf_conversions
from sensor_msgs.msg import Imu

def quat_mul(a, b):
    return tf_conversions.transformations.quaternion_multiply(a, b)

def quat_from_rpy(roll, pitch, yaw):
    return tf_conversions.transformations.quaternion_from_euler(roll, pitch, yaw)

def rotmat_from_quat(q):
    return tf_conversions.transformations.quaternion_matrix(q)[:3, :3]

class ImuTransformer:
    def __init__(self):
        # Frames/params
        self.target_frame = rospy.get_param("~target_frame", "r5000219a8/base_front_mid_laser_link")
        self.rpy_deg = rospy.get_param("~rpy_deg", [90.0, 0.0, 0.0])   # roll, pitch, yaw (deg). Start with +90 roll (Y to Z)
        self.accel_scale = rospy.get_param("~accel_scale", 9.80665)    # g to m/s^2 ; set to 1.0 if already m/s^2
        self.gyro_scale  = rospy.get_param("~gyro_scale", 1.0)         # deg/s  rad/s use np.deg2rad(1.0) if needed

        rr, rp, ry = np.deg2rad(self.rpy_deg)
        self.q_ST = quat_from_rpy(rr, rp, ry)  # source to target rotation
        self.R_ST = rotmat_from_quat(self.q_ST)

        self.sub = rospy.Subscriber("imu_in", Imu, self.cb, queue_size=200)
        self.pub = rospy.Publisher("imu_out", Imu, queue_size=200)

    def cb(self, msg):
        out = Imu()
        out.header = msg.header
        out.header.frame_id = self.target_frame

        # Orientation: if unknown (all zeros), mark as unknown; else rotate: q_T = q_ST * q_S
        qS = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        if np.allclose(qS, 0.0):
            out.orientation.x = out.orientation.y = out.orientation.z = out.orientation.w = 0.0
            cov = list(msg.orientation_covariance)
            if len(cov) == 9:
                cov[0] = -1.0  # REP-145: mark orientation as not provided
            out.orientation_covariance = cov
        else:
            qT = quat_mul(self.q_ST, qS)
            out.orientation.x, out.orientation.y, out.orientation.z, out.orientation.w = qT
            out.orientation_covariance = msg.orientation_covariance

        # Angular velocity (apply rotation & optional unit scale)
        wS = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]) * self.gyro_scale
        wT = self.R_ST.dot(wS)
        out.angular_velocity.x, out.angular_velocity.y, out.angular_velocity.z = wT
        out.angular_velocity_covariance = msg.angular_velocity_covariance

        # Linear acceleration (apply rotation & unit scale)
        aS = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]) * self.accel_scale
        aT = self.R_ST.dot(aS)
        out.linear_acceleration.x, out.linear_acceleration.y, out.linear_acceleration.z = aT
        out.linear_acceleration_covariance = msg.linear_acceleration_covariance

        self.pub.publish(out)

if __name__ == "__main__":
    rospy.init_node("imu_transformer")
    ImuTransformer()
    rospy.spin()
