#pragma once

#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <string>

#include <ros/ros.h>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include "use-ikfom.hpp"
#include <tf/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>

/// ************* Preconfiguration

#define MAX_INI_COUNT (10)

const bool time_list(PointType &x, PointType &y) { return (x.curvature < y.curvature); };

/// ************* IMU Process and undistortion
class ImuProcess
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit ImuProcess(const std::string &rn);
  ~ImuProcess();

  void Reset();
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void set_extrinsic(const V3D &transl, const M3D &rot);
  void set_extrinsic(const V3D &transl);
  void set_extrinsic(const MD(4, 4) &T);
  void set_gyr_cov(const V3D &scaler);
  void set_acc_cov(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  Eigen::Matrix<double, 12, 12> Q;

  void Process(const MeasureGroup &meas,
               esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
               PointCloudXYZI::Ptr pcl_un_);

  void set_node_handler(const ros::Publisher pub) { odom_pub_ = pub; }

  // Optional, if you ever want to change at runtime
  void set_robot_name(const std::string &rn)
  {
    robot_name_ = rn;
    recompute_frame_ids_();
    ROS_INFO_STREAM("robot name set to " << (robot_name_.empty() ? "<global>" : robot_name_));
  }

  void InitializeFromOdom();

  ofstream fout_imu;
  V3D cov_acc;
  V3D cov_gyr;
  V3D cov_acc_scale;
  V3D cov_gyr_scale;
  V3D cov_bias_gyr;
  V3D cov_bias_acc;
  double first_lidar_time;

private:
  void IMU_init(const MeasureGroup &meas,
                esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                int &N);
  void UndistortPcl(const MeasureGroup &meas,
                    esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                    PointCloudXYZI &pcl_in_out);

  // Frame computation: publish to base_link (FAST-LIO pose),
  // but SEED from an existing URDF base frame (seed_base_frame_).
  void recompute_frame_ids_()
  {
    const bool global_ns = robot_name_.empty();
    odom_link        = global_ns ? std::string("odom")              : (robot_name_ + "/odom");
    base_link        = global_ns ? std::string("fast_lio_pose")     : (robot_name_ + "/fast_lio_pose"); // output frame we PUBLISH
    seed_base_frame_ = global_ns ? std::string("base_link")         : (robot_name_ + "/base_link");     // existing frame we READ to seed
  }

  PointCloudXYZI::Ptr cur_pcl_un_;
  sensor_msgs::ImuConstPtr last_imu_;
  deque<sensor_msgs::ImuConstPtr> v_imu_;
  vector<Pose6D> IMUpose;
  vector<M3D> v_rot_pcl_;
  M3D Lidar_R_wrt_IMU;
  V3D Lidar_T_wrt_IMU;
  V3D mean_acc;
  V3D mean_gyr;
  V3D angvel_last;
  V3D acc_s_last;
  std::string odom_link, base_link, seed_base_frame_, robot_name_, ns;

  double start_timestamp_;
  double last_lidar_end_time_;
  int init_iter_num = 1;
  bool b_first_frame_ = true;
  bool imu_need_init_ = true;
  tf::TransformListener *tf_listener_ = nullptr;
  bool initialized_from_odom_ = false;
  Eigen::Vector3d initial_pos_;
  Eigen::Quaterniond initial_rot_;

  ros::Publisher odom_pub_;
  void PublishOdometry(const state_ikfom &imu_state, double timestamp);
};

/// ==================== Implementation ====================

inline ImuProcess::ImuProcess(const std::string &rn)
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1), robot_name_(rn)
{
  init_iter_num = 1;
  Q = process_noise_cov();
  cov_acc = V3D(0.1, 0.1, 0.1);
  cov_gyr = V3D(0.1, 0.1, 0.1);
  cov_bias_gyr = V3D(1e-4, 1e-4, 1e-4);
  cov_bias_acc = V3D(1e-4, 1e-4, 1e-4);
  mean_acc = V3D(0, 0, -1.0);
  mean_gyr = V3D(0, 0, 0);
  angvel_last = Zero3d;
  Lidar_T_wrt_IMU = Zero3d;
  Lidar_R_wrt_IMU = Eye3d;

  recompute_frame_ids_();
  ROS_INFO_STREAM("robot name is " << (robot_name_.empty() ? "<global>" : robot_name_));

  last_imu_.reset(new sensor_msgs::Imu());
}

inline ImuProcess::~ImuProcess() {}

inline void ImuProcess::Reset()
{
  mean_acc = V3D(0, 0, -1.0);
  mean_gyr = V3D(0, 0, 0);
  angvel_last = Zero3d;
  imu_need_init_ = true;
  start_timestamp_ = -1;
  init_iter_num = 1;
  v_imu_.clear();
  IMUpose.clear();
  last_imu_.reset(new sensor_msgs::Imu());
  cur_pcl_un_.reset(new PointCloudXYZI());
}

inline void ImuProcess::set_extrinsic(const MD(4, 4) &T)
{
  Lidar_T_wrt_IMU = T.block<3, 1>(0, 3);
  Lidar_R_wrt_IMU = T.block<3, 3>(0, 0);
}

inline void ImuProcess::InitializeFromOdom()
{
  if (!tf_listener_)
    tf_listener_ = new tf::TransformListener();

  tf::StampedTransform tf_odom_to_seed;
  try
  {
    tf_listener_->waitForTransform(odom_link, seed_base_frame_, ros::Time(0), ros::Duration(3.0));
    tf_listener_->lookupTransform(odom_link, seed_base_frame_, ros::Time(0), tf_odom_to_seed);
    ROS_INFO_STREAM("Seeding FAST-LIO from TF(" << odom_link << " -> " << seed_base_frame_ << ")");

    tf::Vector3 t = tf_odom_to_seed.getOrigin();
    tf::Quaternion q = tf_odom_to_seed.getRotation();
    initial_pos_ = Eigen::Vector3d(t.x(), t.y(), t.z());
    initial_rot_ = Eigen::Quaterniond(q.w(), q.x(), q.y(), q.z());

    initialized_from_odom_ = true;
  }
  catch (tf::TransformException &ex)
  {
    ROS_WARN("Could not get %s->%s: %s", odom_link.c_str(), seed_base_frame_.c_str(), ex.what());
    initialized_from_odom_ = false;
  }
}

inline void ImuProcess::set_extrinsic(const V3D &transl)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU.setIdentity();
}

inline void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
}

inline void ImuProcess::set_gyr_cov(const V3D &scaler) { cov_gyr_scale = scaler; }
inline void ImuProcess::set_acc_cov(const V3D &scaler) { cov_acc_scale = scaler; }
inline void ImuProcess::set_gyr_bias_cov(const V3D &b_g) { cov_bias_gyr = b_g; }
inline void ImuProcess::set_acc_bias_cov(const V3D &b_a) { cov_bias_acc = b_a; }

inline void ImuProcess::IMU_init(const MeasureGroup &meas,
                                 esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                                 int &N)
{
  ROS_DEBUG("ImuProcess::IMU_init");
  V3D cur_acc, cur_gyr;

  if (initialized_from_odom_)
  {
    state_ikfom init_state = kf_state.get_x();
    init_state.pos = initial_pos_;
    init_state.rot = initial_rot_.toRotationMatrix();
    kf_state.change_x(init_state);
    ROS_INFO_STREAM("FAST-LIO initial pose set from TF seed: "
                    << "x=" << initial_pos_.x()
                    << ", y=" << initial_pos_.y());
  }

  if (b_first_frame_)
  {
    Reset();
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    first_lidar_time = meas.lidar_beg_time;
  }

  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc += (cur_acc - mean_acc) / N;
    mean_gyr += (cur_gyr - mean_gyr) / N;

    cov_acc = cov_acc * (N - 1.0) / N +
              (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N +
              (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

    N++;
  }

  state_ikfom init_state = kf_state.get_x();
  init_state.grav = S2(-mean_acc / mean_acc.norm() * G_m_s2);
  init_state.bg = mean_gyr;
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;
  init_state.offset_R_L_I = Lidar_R_wrt_IMU;
  kf_state.change_x(init_state);

  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();
  init_P.setIdentity();
  init_P(6, 6) = init_P(7, 7) = init_P(8, 8) = 0.00001;
  init_P(9, 9) = init_P(10, 10) = init_P(11, 11) = 0.00001;
  init_P(15, 15) = init_P(16, 16) = init_P(17, 17) = 0.0001;
  init_P(18, 18) = init_P(19, 19) = init_P(20, 20) = 0.001;
  init_P(21, 21) = init_P(22, 22) = 0.00001;
  kf_state.change_P(init_P);
  last_imu_ = meas.imu.back();
}

inline void ImuProcess::PublishOdometry(const state_ikfom &imu_state, double timestamp)
{
  nav_msgs::Odometry odom_msg;
  odom_msg.header.stamp = ros::Time().fromSec(timestamp);
  odom_msg.header.frame_id = odom_link;
  odom_msg.child_frame_id = base_link; // FAST-LIO output frame

  geometry_msgs::Pose pose_msg;
  pose_msg.position.x = imu_state.pos.x();
  pose_msg.position.y = imu_state.pos.y();
  pose_msg.position.z = imu_state.pos.z();

  Eigen::Quaterniond q(imu_state.rot);
  pose_msg.orientation.x = q.x();
  pose_msg.orientation.y = q.y();
  pose_msg.orientation.z = q.z();
  pose_msg.orientation.w = q.w();

  odom_msg.pose.pose = pose_msg;
  odom_pub_.publish(odom_msg);

  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion q_tf;
  transform.setOrigin(tf::Vector3(imu_state.pos.x(),
                                  imu_state.pos.y(),
                                  imu_state.pos.z()));
  q_tf.setW(q.w());
  q_tf.setX(q.x());
  q_tf.setY(q.y());
  q_tf.setZ(q.z());
  transform.setRotation(q_tf);
  br.sendTransform(tf::StampedTransform(transform, odom_msg.header.stamp, odom_link, base_link));
}

inline void ImuProcess::UndistortPcl(const MeasureGroup &meas,
                                     esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                                     PointCloudXYZI &pcl_out)
{
  ROS_DEBUG("ImuProcess::UndistortPcl");
  auto v_imu_local = meas.imu;
  v_imu_local.push_front(last_imu_);

  const double &imu_end_time = v_imu_local.back()->header.stamp.toSec();
  const double &pcl_beg_time = meas.lidar_beg_time;
  const double &pcl_end_time = meas.lidar_end_time;

  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);

  state_ikfom imu_state = kf_state.get_x();
  IMUpose.clear();
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));

  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;

  double dt = 0;
  input_ikfom in;

  for (auto it_imu = v_imu_local.begin(); it_imu < (v_imu_local.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);
    if (tail->header.stamp.toSec() < last_lidar_end_time_)
      continue;

    angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
        0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
        0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
        0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
        0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    acc_avr = acc_avr * G_m_s2 / mean_acc.norm();

    if (head->header.stamp.toSec() < last_lidar_end_time_)
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;
    else
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();

    in.acc = acc_avr;
    in.gyro = angvel_avr;
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    kf_state.predict(dt, Q, in);

    imu_state = kf_state.get_x();

    // Do not publish any odometry during initialization.
    if (!imu_need_init_)
    {
      PublishOdometry(imu_state, tail->header.stamp.toSec());
    }

    angvel_last = angvel_avr - imu_state.bg;
    acc_s_last = imu_state.rot * (acc_avr - imu_state.ba);
    for (int i = 0; i < 3; i++)
      acc_s_last[i] += imu_state.grav[i];
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
  }

  const double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  kf_state.predict(dt, Q, in);

  imu_state = kf_state.get_x();
  last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;

  if (pcl_out.points.begin() == pcl_out.points.end())
    return;

  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu << MAT_FROM_ARRAY(head->rot);
    vel_imu << VEC_FROM_ARRAY(head->vel);
    pos_imu << VEC_FROM_ARRAY(head->pos);
    acc_imu << VEC_FROM_ARRAY(tail->acc);
    angvel_avr << VEC_FROM_ARRAY(tail->gyr);

    for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--)
    {
      dt = it_pcl->curvature / double(1000) - head->offset_time;

      M3D R_i(R_imu * Exp(angvel_avr, dt));
      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
      V3D P_compensate = imu_state.offset_R_L_I.conjugate() *
                         (imu_state.rot.conjugate() *
                              (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) -
                          imu_state.offset_T_L_I);

      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin())
        break;
    }
  }
}

inline void ImuProcess::Process(const MeasureGroup &meas,
                                esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                                PointCloudXYZI::Ptr cur_pcl_un_)
{
  ROS_DEBUG("ImuProcess::Process");
  if (meas.imu.empty())
    return;
  ROS_ASSERT(meas.lidar != nullptr);

  if (imu_need_init_)
  {
    InitializeFromOdom();
    IMU_init(meas, kf_state, init_iter_num);

    // keep initializing until MAX_INI_COUNT
    last_imu_ = meas.imu.back();

    if (init_iter_num > MAX_INI_COUNT)
    {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;

      // switch to configured covariances
      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
      ROS_INFO("IMU Initial Done");
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"), ios::out);
    }
    return;
  }

  UndistortPcl(meas, kf_state, *cur_pcl_un_);
}
