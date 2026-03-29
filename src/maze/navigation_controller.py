# navigation_controller.py
"""
Pure pursuit navigation controller.

Converts a list of waypoints into velocity commands [vx, vy, yaw_rate]
compatible with the walking policy's command interface.
"""

import math


class NavigationController:
    """Pure pursuit controller for waypoint following.

    Converts a path of (x, y) waypoints into velocity commands that can
    be fed to the walking policy via fixed_command.
    """

    # Stop-turn-walk thresholds (degrees converted to radians)
    TURN_STOP_THRESHOLD = math.radians(25)   # Stop walking when error exceeds this
    TURN_RESUME_THRESHOLD = math.radians(12) # Resume walking when error drops below this

    def __init__(
        self,
        waypoints,
        target_speed=0.3,
        max_yaw_rate=0.4,
        lookahead_distance=1.5,
        reach_radius=0.5,
        kp_yaw=0.8,
        goal_threshold=0.5,
    ):
        """
        Args:
            waypoints: List of (x, y) world-coordinate waypoints.
            target_speed: Forward speed in m/s (default matches walking Stage 0).
            max_yaw_rate: Maximum yaw rate in rad/s.
            lookahead_distance: How far ahead to look along the path in meters.
            reach_radius: Distance to waypoint before advancing to next.
            kp_yaw: Proportional gain for heading error to yaw rate.
            goal_threshold: Distance to final waypoint to consider goal reached.
        """
        self.waypoints = list(waypoints)
        self.target_speed = target_speed
        self.max_yaw_rate = max_yaw_rate
        self.lookahead_distance = lookahead_distance
        self.reach_radius = reach_radius
        self.kp_yaw = kp_yaw
        self.goal_threshold = goal_threshold

        self.current_waypoint_idx = 0
        self._goal_reached = False
        self._committed_turn_dir = None  # Hysteresis for large heading errors
        self._turning_in_place = False   # Stop-turn-walk state

    @property
    def goal_reached(self):
        """Whether the final waypoint has been reached."""
        return self._goal_reached

    def get_command(self, position, heading):
        """Compute velocity command for the current state.

        The walking policy tracks world-frame velocity commands (vx, vy) and
        a yaw rate. We command velocity toward the WAYPOINT (desired heading)
        so the command changes slowly — matching training where commands are
        held for entire episodes. The yaw rate steers the robot to face the
        waypoint direction.

        Args:
            position: (x, y) current position in world frame.
            heading: Current heading angle in radians (0 = +x axis).

        Returns:
            (vx, vy, yaw_rate) velocity command tuple in world frame.
        """
        if self._goal_reached or len(self.waypoints) == 0:
            return (0.0, 0.0, 0.0)

        px, py = position

        # Advance waypoint if within reach
        while self.current_waypoint_idx < len(self.waypoints) - 1:
            wx, wy = self.waypoints[self.current_waypoint_idx]
            dist = math.sqrt((px - wx) ** 2 + (py - wy) ** 2)
            if dist < self.reach_radius:
                self.current_waypoint_idx += 1
            else:
                break

        # Check goal reached
        gx, gy = self.waypoints[-1]
        goal_dist = math.sqrt((px - gx) ** 2 + (py - gy) ** 2)
        if goal_dist < self.goal_threshold:
            self._goal_reached = True
            return (0.0, 0.0, 0.0)

        # Find lookahead point
        target = self._find_lookahead(position)
        tx, ty = target

        # Compute heading error
        desired_heading = math.atan2(ty - py, tx - px)
        heading_error = self._normalize_angle(desired_heading - heading)
        abs_error = abs(heading_error)

        # --- Stop-turn-walk with hysteresis ---
        # The policy can walk forward or stand+turn, but NOT both at once
        # for large heading errors. Use hysteresis to avoid oscillating.
        if abs_error > self.TURN_STOP_THRESHOLD:
            self._turning_in_place = True
        elif abs_error < self.TURN_RESUME_THRESHOLD:
            self._turning_in_place = False

        # Yaw rate control with hysteresis for near-180° turns.
        if abs_error > math.pi * 0.75:
            if self._committed_turn_dir is None:
                self._committed_turn_dir = 1.0 if heading_error > 0 else -1.0
            yaw_rate = self._committed_turn_dir * self.max_yaw_rate
        else:
            self._committed_turn_dir = None
            yaw_rate = self.kp_yaw * heading_error
            yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, yaw_rate))

        if self._turning_in_place:
            # Stand still, only turn — the policy handles (0, 0, yaw) well
            return (0.0, 0.0, yaw_rate)

        # Walking phase — heading is roughly aligned, walk toward waypoint
        speed = self.target_speed * math.cos(heading_error)

        # Velocity toward the WAYPOINT (desired heading) in world frame.
        vx = speed * math.cos(desired_heading)
        vy = speed * math.sin(desired_heading)

        return (vx, vy, yaw_rate)

    def _find_lookahead(self, position):
        """Find the lookahead point along the path.

        Interpolates along the remaining path to find a point at
        lookahead_distance ahead of the current position.

        Args:
            position: (x, y) current position.

        Returns:
            (x, y) lookahead target point.
        """
        px, py = position
        remaining_dist = self.lookahead_distance

        for i in range(self.current_waypoint_idx, len(self.waypoints) - 1):
            ax, ay = self.waypoints[i]
            bx, by = self.waypoints[i + 1]
            seg_len = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)

            if seg_len < 1e-6:
                continue

            if remaining_dist <= seg_len:
                # Interpolate along this segment
                t = remaining_dist / seg_len
                return (ax + t * (bx - ax), ay + t * (by - ay))

            remaining_dist -= seg_len

        # Past end of path — return final waypoint
        return self.waypoints[-1]

    @staticmethod
    def _normalize_angle(angle):
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def reset(self, waypoints=None):
        """Reset the controller, optionally with new waypoints.

        Args:
            waypoints: New list of (x, y) waypoints, or None to reuse existing.
        """
        if waypoints is not None:
            self.waypoints = list(waypoints)
        self.current_waypoint_idx = 0
        self._goal_reached = False
        self._committed_turn_dir = None
        self._turning_in_place = False
