# navigation_controller.py
"""
Pure pursuit navigation controller.

Converts a list of waypoints into velocity commands [vx, vy, yaw_rate]
compatible with the walking policy's command interface.

Strategy: stop-turn-walk. For large heading errors (>60°), the robot stops
and turns in place. For moderate errors (30-60°), it slows down while turning.
For small errors (<30°), it walks at full speed. Velocity is always commanded
along the robot's current heading — matching the training distribution of
forward walking + yaw commands.
"""

import math


class NavigationController:
    """Stop-turn-walk controller for waypoint following.

    Commands velocity along the robot's CURRENT heading (body-forward
    projected into world frame). For large heading errors, stops and turns
    in place. Yaw rate steers the robot to face each waypoint.
    """

    def __init__(
        self,
        waypoints,
        target_speed=0.3,
        max_yaw_rate=0.5,
        lookahead_distance=1.5,
        reach_radius=1.0,
        kp_yaw=1.0,
        goal_threshold=1.0,
    ):
        """
        Args:
            waypoints: List of (x, y) world-coordinate waypoints.
            target_speed: Forward speed in m/s.
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

        Commands velocity along robot's current heading. Stops for large
        heading errors to turn in place first.

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

        # Yaw correction — proportional control, clamped
        yaw_rate = self.kp_yaw * heading_error
        yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, yaw_rate))

        # Command velocity along ROBOT HEADING (body-forward), not toward waypoint.
        # The policy was trained on forward walking + yaw — not lateral walking.
        # Large heading errors: slow down and turn in place first.
        abs_err = abs(heading_error)
        if abs_err > math.pi / 3:
            # Large error (>60°): stop and turn in place
            speed = 0.0
        elif abs_err > math.pi / 6:
            # Moderate error (30-60°): slow down while turning
            speed = self.target_speed * (1.0 - (abs_err - math.pi / 6) / (math.pi / 6))
        else:
            # Small error (<30°): full speed
            speed = self.target_speed

        vx = speed * math.cos(heading)
        vy = speed * math.sin(heading)

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
