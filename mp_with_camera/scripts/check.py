#!/usr/bin/env python3
import rospy, numpy as np, csv, os
from visualization_msgs.msg import Marker
from std_msgs.msg import Float64MultiArray, Bool, String, Header

class HumanRobotSafetySystem:
    def __init__(self):
        rospy.init_node('safety', anonymous=True)

        # ───────── 파라미터 (기본값) ─────────
        self.emergency_distance = rospy.get_param('~emergency_distance', 0.90)  # STOP
        self.release_distance   = rospy.get_param('~release_distance',   1.00)  # CLEAR
        self.warning_distance   = rospy.get_param('~warning_distance',   1.00)  # WARNING

        # sanity-check: release > emergency
        if not self.emergency_distance < self.release_distance:
            raise ValueError('~release_distance must be larger than ~emergency_distance')

        # ───────── 토픽 I/O ─────────
        rospy.Subscriber('/visualization_marker', Marker,
                         self.human_cb,  queue_size=1)
        rospy.Subscriber('/panda_csv', Float64MultiArray,
                         self.robot_cb,  queue_size=1)

        self.emerg_pub  = rospy.Publisher('/emergency_stop',   Bool,   queue_size=1)
        self.warn_pub   = rospy.Publisher('/safety_warning',   Bool,   queue_size=1)
        self.status_pub = rospy.Publisher('/safety_status',    String, queue_size=1)
        self.stamp_pub  = rospy.Publisher('/emergency_timestamp', Header, queue_size=1)

        self.human_pos = None
        self.robot_pos = None
        self.emergency_active = False

        self.last_dist = None
        self.print_cnt = 0

        # CSV 설정
        self.csv_file = rospy.get_param('~csv_file', 'distance_log.csv')
        self._init_csv()

        rospy.Timer(rospy.Duration(0.02), self.timer_cb)       # 50 Hz

    # ───────── 콜백 ─────────
    def human_cb(self, msg):
        self.human_pos = np.array([msg.pose.position.x,
                                   msg.pose.position.y,
                                   msg.pose.position.z], dtype=float)

    def robot_cb(self, msg):
        if len(msg.data) >= 3:
            self.robot_pos = np.array(msg.data[:3], dtype=float)

    # ───────── 거리 계산 ─────────
    @staticmethod
    def distance(a, b):
        return np.linalg.norm(a - b)

    # ───────── 주기적 확인 ─────────
    def timer_cb(self, _):
        if self.human_pos is None or self.robot_pos is None:
            return

        d = self.distance(self.human_pos, self.robot_pos)

        # CSV 기록
        self._save_distance_csv(d)

        # 콘솔 출력(변동 5 cm 이상 또는 0.5 s마다)
        self._print_distance_once(d)

        # ── EMERGENCY 진입 ───────────
        if not self.emergency_active and d <= self.emergency_distance:
            self._enter_emergency(d)

        # ── EMERGENCY 해제 ───────────
        elif self.emergency_active and d >= self.release_distance:
            self._exit_emergency(d)

        # ── WARNING 플래그 ───────────
        warn = (d < self.warning_distance)
        self.warn_pub.publish(warn and not self.emergency_active)
        self.status_pub.publish('EMERGENCY_STOP' if self.emergency_active
                                else ('WARNING' if warn else 'SAFE'))

    # ───────── 상태 전환 ─────────
    def _enter_emergency(self, dist):
        rospy.logerr(f'EMERGENCY_STOP  d={dist:.3f} m')
        self.emergency_active = True
        self.emerg_pub.publish(True)

        hd = Header()
        hd.stamp = rospy.Time.now()
        self.stamp_pub.publish(hd)

    def _exit_emergency(self, dist):
        rospy.loginfo(f'EMERGENCY_CLEAR d={dist:.3f} m')
        self.emergency_active = False
        self.emerg_pub.publish(False)

    # ───────── CSV 관련 ─────────
    def _init_csv(self):
        if not os.path.isfile(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as f:
                csv.writer(f).writerow(['timestamp_sec', 'distance_m'])

    def _save_distance_csv(self, distance):
        """현재 ROS 시간(sec)과 거리를 CSV로 추가 저장."""
        t = rospy.Time.now().to_sec()    # float seconds[2][4]
        with open(self.csv_file, mode='a', newline='') as f:
            csv.writer(f).writerow([f'{t:.9f}', f'{distance:.6f}'])

    # ───────── 출력 ─────────
    def _print_distance_once(self, dist):
        ch = (self.last_dist is None) or abs(dist - self.last_dist) > 0.05
        self.print_cnt += 1
        if ch or self.print_cnt >= 25:
            print(f'[SAFETY] distance={dist:.3f} m '
                  f'(stop {self.emergency_distance}, clear {self.release_distance})')
            self.print_cnt = 0
            self.last_dist = dist

if __name__ == '__main__':
    try:
        HumanRobotSafetySystem()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

