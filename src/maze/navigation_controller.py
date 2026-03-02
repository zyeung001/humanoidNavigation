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

    def __init__(
        self,
        waypoints,
        target_speed=0.3,
        max_yaw_rate=1.0,
        lookahead_distance=1.5,
        reach_radius=0.5,
        kp_yaw=2.0,
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

    @property
    def goal_reached(self):
        """Whether the final waypoint has been reached."""
        return self._goal_reached

    def get_command(self, position, heading):
        """Compute velocity command for the current state.

        Args:
            position: (x, y) current position in world frame.
            heading: Current heading angle in radians (0 = +x axis).

        Returns:
            (vx, vy, yaw_rate) velocity command tuple.
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

        # Proportional yaw rate control
        yaw_rate = self.kp_yaw * heading_error
        yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, yaw_rate))

        # Speed modulation — slow down for sharp turns
        speed = self.target_speed * max(0.0, math.cos(heading_error))

        # Velocity in world frame, decomposed into forward/lateral
        vx = speed * math.cos(heading_error)
        vy = speed * math.sin(heading_error)

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
