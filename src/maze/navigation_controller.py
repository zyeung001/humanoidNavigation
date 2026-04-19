# navigation_controller.py
"""
Pure pursuit navigation controller.

Converts a list of waypoints into velocity commands [vx, vy, yaw_rate]
compatible with the walking policy's command interface.

Strategy: walk-and-steer. Always maintains minimum forward speed while turning.
The humanoid was trained on forward walking + yaw commands — it cannot stop
and turn in place (17M steps of forward walking too ingrained). For large
heading errors, slow down but keep walking. The robot turns in a curve.
"""

import math


class NavigationController:
    """Walk-and-steer controller for waypoint following.

    Always maintains minimum forward speed while turning. The humanoid
    can't stop walking (forward gait too ingrained from 17M steps of training),
    so we turn in curves instead of turning in place.
    """

    def __init__(
        self,
        waypoints,
        target_speed=0.3,
        min_speed=0.10,
        max_yaw_rate=0.5,
        lookahead_distance=1.5,
        reach_radius=1.0,
        kp_yaw=2.0,
        goal_threshold=1.0,
    ):
        """
        Args:
            waypoints: List of (x, y) world-coordinate waypoints.
            target_speed: Forward speed in m/s.
            min_speed: Minimum forward speed (never stop — agent can't turn in place).
            max_yaw_rate: Maximum yaw rate in rad/s.
            lookahead_distance: How far ahead to look along the path in meters.
            reach_radius: Distance to waypoint before advancing to next.
            kp_yaw: Proportional gain for heading error to yaw rate.
            goal_threshold: Distance to final waypoint to consider goal reached.
        """
        self.waypoints = list(waypoints)
        self.target_speed = target_speed
        self.min_speed = min_speed
        self.max_yaw_rate = max_yaw_rate
        self.lookahead_distance = lookahead_distance
        self.reach_radius = reach_radius
        self.kp_yaw = kp_yaw
        self.goal_threshold = goal_threshold

        self.current_waypoint_idx = 0
        self._goal_reached = False
        self._tip_active = False
        self._frozen_desired_heading = None
        self._brake_counter = 0

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

        # Compute desired heading from lookahead
        desired_heading = math.atan2(ty - py, tx - px)

        # Stop-and-turn: if heading error is large, halt and rotate in place.
        # Otherwise walk forward along current heading. Requires TIP-trained
        # policy (turn_in_place_prob > 0) to execute (0, 0, yaw) cleanly.
        #
        # When TIP engages, FREEZE the desired heading — micro-motions during
        # rotation shift the lookahead target, causing the desired heading to
        # oscillate and preventing the agent from ever fully aligning. The
        # frozen target releases only after heading error drops below the
        # resume threshold.
        stop_threshold = math.radians(45)
        resume_threshold = math.radians(20)

        if self._tip_active and self._frozen_desired_heading is not None:
            active_desired = self._frozen_desired_heading
        else:
            active_desired = desired_heading

        heading_error = self._normalize_angle(active_desired - heading)
        abs_err = abs(heading_error)

        yaw_rate = self.kp_yaw * heading_error
        yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, yaw_rate))

        if self._tip_active:
            if abs_err <= resume_threshold:
                self._tip_active = False
                self._frozen_desired_heading = None
                self._brake_counter = 0
            else:
                # Brake phase: first N steps of TIP command reverse velocity to
                # kill forward momentum before rotating. Without this, the agent
                # keeps coasting forward from the prior walking phase and can
                # collide with a wall at sharp corners.
                if self._brake_counter < 40:
                    self._brake_counter += 1
                    brake_vx = -0.10 * math.cos(heading)
                    brake_vy = -0.10 * math.sin(heading)
                    return (brake_vx, brake_vy, yaw_rate)
                return (0.0, 0.0, yaw_rate)
        elif abs_err > stop_threshold:
            self._tip_active = True
            self._frozen_desired_heading = desired_heading
            self._brake_counter = 0
            return (0.0, 0.0, yaw_rate)

        if abs_err > resume_threshold:
            t = (abs_err - resume_threshold) / (stop_threshold - resume_threshold)
            speed = self.target_speed * (1.0 - t) + self.min_speed * t
        else:
            speed = self.target_speed

        # Walking mode: command velocity toward LOOKAHEAD TARGET plus
        # cross-track correction. The policy has residual forward bias that
        # accumulates lateral drift over long corridors — without active
        # correction the agent drifts into walls. We compute the closest
        # point on the path (projection) and add a pull toward it.
        proj_x, proj_y = self._find_projection(position)
        ct_dx = proj_x - px
        ct_dy = proj_y - py
        ct_err = math.sqrt(ct_dx * ct_dx + ct_dy * ct_dy)

        dx = tx - px
        dy = ty - py
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-6:
            return (0.0, 0.0, yaw_rate)

        # Unit vector toward lookahead target
        ux = dx / dist
        uy = dy / dist

        # Cross-track correction: pull toward projection point. Strength scales
        # with error, capped so correction never exceeds forward speed.
        ct_gain = 1.0
        ct_mag = min(ct_gain * ct_err, 0.7 * speed)
        if ct_err > 1e-6:
            cx = ct_mag * ct_dx / ct_err
            cy = ct_mag * ct_dy / ct_err
        else:
            cx = cy = 0.0

        vx = speed * ux + cx
        vy = speed * uy + cy

        return (vx, vy, yaw_rate)

    def _find_projection(self, position):
        """Find the closest point on the path from current waypoint onward.

        Returns (x, y) of the projection of `position` onto the path.
        """
        px, py = position
        best = (px, py)
        best_dist_sq = float('inf')
        for i in range(max(self.current_waypoint_idx - 1, 0),
                       len(self.waypoints) - 1):
            ax, ay = self.waypoints[i]
            bx, by = self.waypoints[i + 1]
            dx, dy = bx - ax, by - ay
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-12:
                continue
            t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            proj_x = ax + t * dx
            proj_y = ay + t * dy
            d_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2
            if d_sq < best_dist_sq:
                best_dist_sq = d_sq
                best = (proj_x, proj_y)
        return best

    def _find_lookahead(self, position):
        """Find the lookahead point along the path via pure pursuit.

        Projects the agent onto the nearest path segment, then walks
        lookahead_distance forward along the remaining path.

        Args:
            position: (x, y) current position.

        Returns:
            (x, y) lookahead target point.
        """
        px, py = position

        # Step 1: Find the closest point on the path from current_waypoint_idx onward
        best_seg = self.current_waypoint_idx
        best_t = 0.0
        best_dist_sq = float('inf')

        for i in range(max(self.current_waypoint_idx - 1, 0), len(self.waypoints) - 1):
            ax, ay = self.waypoints[i]
            bx, by = self.waypoints[i + 1]
            dx, dy = bx - ax, by - ay
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-12:
                continue
            # Project position onto segment
            t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            proj_x = ax + t * dx
            proj_y = ay + t * dy
            dist_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_seg = i
                best_t = t

        # Step 2: Walk lookahead_distance forward from the projection point
        seg_idx = best_seg
        ax, ay = self.waypoints[seg_idx]
        bx, by = self.waypoints[seg_idx + 1] if seg_idx + 1 < len(self.waypoints) else (ax, ay)
        dx, dy = bx - ax, by - ay
        seg_len = math.sqrt(dx * dx + dy * dy)

        # Remaining distance on the current segment after the projection
        remaining_on_seg = seg_len * (1.0 - best_t)
        remaining_dist = self.lookahead_distance

        if remaining_dist <= remaining_on_seg:
            # Lookahead falls on the current segment
            t_final = best_t + remaining_dist / max(seg_len, 1e-6)
            return (ax + t_final * dx, ay + t_final * dy)

        remaining_dist -= remaining_on_seg

        # Walk through subsequent segments
        for i in range(seg_idx + 1, len(self.waypoints) - 1):
            sx, sy = self.waypoints[i]
            ex, ey = self.waypoints[i + 1]
            s_len = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
            if s_len < 1e-6:
                continue
            if remaining_dist <= s_len:
                t = remaining_dist / s_len
                return (sx + t * (ex - sx), sy + t * (ey - sy))
            remaining_dist -= s_len

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
        self._tip_active = False
        self._frozen_desired_heading = None
        self._brake_counter = 0
