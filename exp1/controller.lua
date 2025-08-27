-- polygons
local edge_length = 30
local polygon_sides = 3

local demo_repeat_target = 5


local state_demo = "signal_start"
local state_learner = "waiting"
local step_demo = 0
local demo_repeat_count = 0

local START_TIME = 600
local approach_distance = 100
local step_count = 0
local state_polygon = "forward"
local step_count_polygon = 0
local side_count_polygon = 0
local complete_polygon = false

local last_angle = 0

local is_exist_segments_demo = false

role = "idle"
local is_following = false

EPSILON = 50 
WHEEL_SPEED = 8 -- max wheel speed

repulsion_gain = 2
repulsion_sensitivity = 1.5
proximity_threshold = 0.05

-- robots
local velocity = 10
local ticks_per_second = 10
local wheel_base = 14.0

local edge_steps = edge_length / velocity * ticks_per_second

local turn_speed =14.65
local trajectory_learner = {}
local trajectory_demo = {}

-- local record_delay = 200
local observation_positions = {}
local segments_demo = {}
local segments_learner = {}
local segments_observed = {}
local current_segment = 1
local segment_step = 0
local imitation_phase_active = false
local imitation_state = "turn"
local turn_step = 0
local learner_done = false

local teacher_quality_scores = {} -- { [teacher_id] = {qualitys = {...}, avg = value} }
local current_teacher_id = nil
local observation_repeat = 0
local max_observation_repeat = 10
local visited_teachers = {} -- set to record visited teacher ids

function init()
    robot.range_and_bearing.clear_data()

    if robot.id == "demo" then
        role = "teacher"
        robot.leds.set_all_colors("green")
        robot.range_and_bearing.set_data(1, 99) 

    elseif robot.id == "learner" then
        role = "learner"
        robot.leds.set_all_colors("yellow")
    else
        role = "idle"
    end
    robot.colored_blob_omnidirectional_camera.enable()
end

function step()
    step_count = step_count + 1

    if role == "idle" then
        RandomWalk()

    elseif role == "teacher" then
        robot.range_and_bearing.set_data(1, 99) 

        step_demo = step_demo + 1
        if state_demo == "signal_start" then
            robot.leds.set_all_colors("green")
            if step_demo >= START_TIME then
                state_demo = "moving"
                step_demo = 0
            end
        elseif state_demo == "moving" then
            PolygonMotion(polygon_sides)
            if robot.id == "demo" then
                -- log("[DEBUG] Robot is the demonstrator, executing polygon motion.")
                --    PolygonMotion()
            else 
                -- log("[DEBUG] Robot is not the demonstrator, skipping polygon motion.")
                -- ImitateTrajectory()
            end

            if complete_polygon then
                step_demo = 0
                state_demo = "signal_end"
            end

        elseif state_demo == "signal_end" then
            robot.leds.set_all_colors("red")
            if step_demo >= 50 then

                state_demo = "done"
            end
        elseif state_demo == "done" then
            robot.wheels.set_velocity(0, 0)
            SaveTrajectoryToFile("trajectory_demo.txt", trajectory_demo)
            -- SaveTrajectoryToFile("trajectory_" .. robot.id .. ".txt", trajectory_demo)
            --  role = "learner"
            --  robot.leds.set_all_colors("yellow")
            --  state_learner = "waiting"

        end

    elseif role == "learner" then
        -- log(string.format("\n[DEBUG] Current state: %s\n", state_learner))

        if state_learner == "waiting" then
            robot.leds.set_all_colors("yellow")
            RandomWalk()
            local closest_rb = nil

            for i = 1, #robot.range_and_bearing do
                local msg = robot.range_and_bearing[i]
                if msg.range <= approach_distance and  LedIsDetected("green") then
                    closest_rb = msg
                end
            end

            if closest_rb then
                -- visit = visit + 1
                state_learner = "recording"
            end

        end

        if state_learner == "recording" then
            
            robot.leds.set_all_colors("yellow")
            -- ApproachDemonstrator()
            FollowDemonstrator()
            if not is_following then
                RecordObservation()
            end
            if LedIsDetected("red") then
                state_learner = "analyzing"
            end
        end
        if state_learner == "analyzing" then
            robot.leds.set_all_colors("yellow")
            SaveTrajectoryToFile("trajectory_observed.txt", observation_positions)
            local filtered_points = FilterOutliers(observation_positions, 0.5, 2)
            SaveTrajectoryToFile("trajectory_filtered.txt", filtered_points)
            
            local largest_cluster = find_largest_cluster(filtered_points, 2.0)
            SaveTrajectoryToFile("trajectory_largest_cluster.txt", largest_cluster)

            local main_path = ExtractMainPathUnique(largest_cluster, 0.2, 1, 0.2, 1)
            main_path = sort_by_nearest_neighbor(main_path)
            main_path = RemoveDuplicatePoints(main_path, 0.01)
            SaveTrajectoryToFile("trajectory_main.txt", main_path)

            main_path = FilterOutliers(main_path, 1.5, 2)

            -- main_path = remove_step_jumps(main_path, 3.0)
            SaveTrajectoryToFile("trajectory_cleaned.txt", main_path)

            local corners = DetectTurnPointsAngle(main_path, 20.0)
            SaveCornersToFile(main_path, corners, "corners_detected.txt")

            segments_observed = LinerRegression(main_path)
            for i, seg in ipairs(segments_observed) do
                log(string.format("[O_SEG_main %d] angle = %.4f, distance = %.2f", i, math.deg(seg.angle), seg.distance))
            end

            state_learner = "imitating"
        end

        if state_learner == "imitating" then
            robot.leds.set_all_colors("cyan")
            ImitateTrajectory()
            SaveTrajectoryToFile("trajectory_learner.txt", trajectory_learner)
        end
        if state_learner == "done" then
            segments_demo = LoadDemoSegments()
            for i, seg in ipairs(segments_demo) do
                log(string.format("[D_SEG %d] angle = %.4f, distance = %.2f", i, math.deg(seg.angle), seg.distance))
            end
            log(string.format("[DEBUG] Imitation completed, segments_demo contains %d segments", #segments_demo))
            trajectory_learner = RemoveDuplicatePoints(trajectory_learner, 0.01)
            segments_learner = LinerRegression(trajectory_learner)

            for i, seg in ipairs(segments_learner) do
                log(string.format("[L_SEG %d] angle = %.4f, distance = %.2f", i, math.deg(seg.angle), seg.distance))
            end
            local quality = ComputeImitationQuality(segments_demo, segments_learner)

            -- log(string.format("[debug] segments_learner contents:\n", segments_learner))
            local f = io.open("quality_scores_4m4_10robots_triangle.txt", "a")
            f:write(string.format("%.4f\n", quality))
            f:close()

            state_learner = "finish"
            -- role = "teacher"
            -- state_demo = "signal_start"
            -- robot.leds.set_all_colors("green")
            -- destroy()
        end
        if state_learner == "finish" then
            robot.wheels.set_velocity(0, 0)
            robot.leds.set_all_colors("red")
        end

    end

end

function SaveObservationToFile()
    local f = io.open("trajectory_observed.txt", "w")
    for i, p in ipairs(observation_positions) do
        f:write(string.format("%.4f,%.4f\n", p.x, p.y))
    end
    f:close()
end

function LoadObservedTrajectory()
    observation_positions = {}
    local f = io.open("trajectory_observed.txt", "r")
    for line in f:lines() do
        local x, y = line:match("([%d.-]+),([%d.-]+)")
        if x and y then
            table.insert(observation_positions, {
                x = tonumber(x),
                y = tonumber(y)
            })
        end
    end
    f:close()
end

function LoadDemoSegments()
    local segments = {}
    local f = io.open("segments_demo.txt", "r")
    if not f then
        log("[ERROR] segments_demo.txt not found, using empty segments.")
        return segments
    end
    for line in f:lines() do
        local angle, distance = line:match("([%d.-]+),([%d.-]+)")
        if angle and distance then
            table.insert(segments, {
                angle = tonumber(angle),
                distance = tonumber(distance)
            })
        end
    end
    f:close()
    return segments
end

function SaveSegmentsToFile(filename, segments)
    local f = io.open(filename, "w")
    if not f then
        log(string.format("[ERROR] Cannot open %s to write segments", filename))
        return
    end
    for i, seg in ipairs(segments) do
        f:write(string.format("%.4f,%.2f\n", seg.angle, seg.distance))
    end
    f:close()
    log(string.format("[INFO] Saved %d segments to %s", #segments, filename))
end

function SaveTrajectoryToFile(filename, trajectory)
    local f = io.open(filename, "w")
    if not f then
        log(string.format("Error: Failed to open %s for writing\n", filename))
        return
    end
    for _, p in ipairs(trajectory) do
        f:write(string.format("%.4f,%.4f\n", p.x, p.y))
    end
    f:close()
end

function SaveCornersToFile(points, corner_indices, filename)
    local f = io.open(filename, "w")
    if not f then
        log(string.format("[ERROR] Cannot open %s to write corners", filename))
        return
    end

    for _, idx in ipairs(corner_indices) do
        local p = points[idx]
        if p then
            f:write(string.format("%.4f,%.4f\n", p.x, p.y))
        end
    end

    f:close()
    log(string.format("[INFO] Saved %d corners to %s", #corner_indices, filename))
end

function destroy()

end

function PolygonMotion(polygon_sides)
    local turn_angle = 2 * math.pi / polygon_sides
    local turn_radius = wheel_base / 2
    local turn_arc = turn_angle * turn_radius
    local turn_speed = turn_arc
    local turn_steps = turn_arc / turn_speed * ticks_per_second
    if complete_polygon then
        robot.wheels.set_velocity(0, 0)
        return
    end

    if IsObstacleAhead(0.2) then
        robot.wheels.set_velocity(0, 0) 
        return
    end

    local pos = robot.positioning.position
    -- log(string.format("[DEBUG] DEMO Current position: x = %.2f, y = %.2f\n", pos.x * 100, pos.y * 100))
    table.insert(trajectory_demo, {
        x = pos.x * 100,
        y = pos.y * 100
    })

    step_count_polygon = step_count_polygon + 1
    if state_polygon == "forward" then

        robot.wheels.set_velocity(velocity, velocity)

        if step_count_polygon >= edge_steps then
            state_polygon = "turn"
            step_count_polygon = 0
        end
    else

        robot.wheels.set_velocity(turn_speed, -turn_speed)

        if step_count_polygon >= turn_steps then
            state_polygon = "forward"
            step_count_polygon = 0
            side_count_polygon = side_count_polygon + 1
            if side_count_polygon >= polygon_sides then
                demo_repeat_count = demo_repeat_count + 1
                log(string.format("[DEBUG] Completed polygon %d\n", demo_repeat_count))
                if demo_repeat_count < demo_repeat_target then
                    side_count_polygon = 0
                    step_count_polygon = 0
                    state_polygon = "forward"
                else
                    complete_polygon = true
                    robot.wheels.set_velocity(0, 0)
                    log("[DEBUG] Completed all polygons, stopping robot.")
                end
                if not is_exist_segments_demo then
                    segments_demo = LinerRegression(trajectory_demo)
                    log(string.format("[DEBUG] Detected %d segments in demonstration trajectory\n", #segments_demo))
                    SaveSegmentsToFile("segments_demo.txt", segments_demo)
                    is_exist_segments_demo = true
                end

            end
        end
    end
end

function RecordObservation()
    for i = 1, #robot.range_and_bearing do
        local msg = robot.range_and_bearing[i]
        if msg.range < 130.0 and msg.data[1] == 99 then
            local b = msg.horizontal_bearing
            local r = msg.range
            local x = r * math.cos(b)
            local y = r * math.sin(b)
            --  log("[DEBUG] Observation: x = %.2f, y = %.2f\n", x, y)
            table.insert(observation_positions, {
                x = x,
                y = y
            })
        end
    end
end



function DetectTurnPoints(points, angle_threshold_deg, window)
    local result = {}
    local angle_threshold = math.rad(angle_threshold_deg)
    local tolerance = 0.05 
    local i = window + 1

    while i <= #points - window - 1 do
        local moving = false
        for j = -window, window - 1 do
            local p1 = points[i + j]
            local p2 = points[i + j + 1]
            if not p1 or not p2 then
                moving = true
                break
            end
            if point_distance(p1, p2) > tolerance then
                moving = true
                break
            end
        end

        if not moving then
            local block = find_repeated_block(points, i, tolerance)

            local start_idx = block[1]
            local end_idx = block[#block]
            local center_idx = math.floor((start_idx + end_idx) / 2)
            local p_mid = points[center_idx]

            local p_prev = nil
            for offset = 1, window do
                local idx = start_idx - offset
                if idx >= 1 and point_distance(points[idx], p_mid) > tolerance then
                    p_prev = points[idx]
                    break
                end
            end

    
            local p_next = nil
            for offset = 1, window do
                local idx = end_idx + offset
                if idx <= #points and point_distance(points[idx], p_mid) > tolerance then
                    p_next = points[idx]
                    break
                end
            end

            if p_prev and p_next then
                local dx1 = p_mid.x - p_prev.x
                local dy1 = p_mid.y - p_prev.y
                local dx2 = p_next.x - p_mid.x
                local dy2 = p_next.y - p_mid.y

                local angle1 = math.atan2(dy1, dx1)
                local angle2 = math.atan2(dy2, dx2)
                local delta = math.abs(angle_diff(angle1, angle2))

                if delta > angle_threshold then
                    table.insert(result, center_idx)
                    log(string.format("[DEBUG] Detected turn point at index %d: (%.2f, %.2f), angle change = %.2f°",
                        center_idx, p_mid.x, p_mid.y, math.deg(delta)))
                end
            end

            i = end_idx + 1
        else
            i = i + 1
        end
    end

    return result
end

function DetectTurnPointsAngle(points, angle_threshold_deg)
    local angle_threshold = math.rad(angle_threshold_deg)
    local result = {}

    for i = 2, #points - 1 do
        local p_prev = points[i - 1]
        local p_curr = points[i]
        local p_next = points[i + 1]

        local angle1 = ComputeAngle(p_prev, p_curr)
        local angle2 = ComputeAngle(p_curr, p_next)

        local diff = math.abs(NormalizeAngleDifference(angle1, angle2))

        if diff >= angle_threshold then
            table.insert(result, i) 
        end
    end

    return result
end

function LinerRegression(points)
    local segments = {}
    local corners = DetectTurnPointsAngle(points, 20.0)
    for i = 1, #corners + 1 do
        local start_idx = (i == 1) and 1 or corners[i - 1]
        local end_idx = corners[i] or #points
        if end_idx - start_idx >= 1 then
            local p1 = points[start_idx]
            local p2 = points[end_idx]
            local dx = p2.x - p1.x
            local dy = p2.y - p1.y
            local angle = math.atan2(dy, dx)
            local distance = math.sqrt(dx * dx + dy * dy)

            if distance > 3.0 then
                --  log(string.format("Segment %d: angle = %.4f rad (%.1f°), distance = %.2f cm\n",
                --  i, angle, math.deg(angle), distance))
                table.insert(segments, {
                    angle = angle,
                    distance = distance
                })
            end
        end
    end

    -- log(string.format("[DEBUG] Detected %d corner points, resulting in %d segments\n", #corners, #segments))
    return segments
end

function angle_diff(target, current)
    local diff = target - current
    while diff > math.pi do
        diff = diff - 2 * math.pi
    end
    while diff < -math.pi do
        diff = diff + 2 * math.pi
    end
    return diff
end

function ImitateTrajectory()
    -- log(string.format("[DEBUG] Imitation state: %s, current segment: %d\n", imitation_state, current_segment))
    local seg = segments_observed[current_segment]
    local pos = robot.positioning.position
    table.insert(trajectory_learner, {
        x = pos.x * 100,
        y = pos.y * 100
    }) 
    if not seg then
        current_segment = 1
        robot.wheels.set_velocity(0, 0)
        state_learner = "done"
        learner_done = true
        complete_polygon = true

        return
    end

    if IsObstacleAhead(0.1) then
        robot.wheels.set_velocity(0, 0) 
        return
    end

    if imitation_state == "turn" then
        local delta_angle = angle_diff(seg.angle, last_angle)
        local target_angle = seg.angle
        local turn_ticks = math.abs(delta_angle) * (wheel_base / 2) / turn_speed * ticks_per_second
        if turn_step < turn_ticks then
            if delta_angle > 0 then
                robot.wheels.set_velocity(-turn_speed, turn_speed)
            else
                robot.wheels.set_velocity(turn_speed, -turn_speed)
            end
            turn_step = turn_step + 1
        else
            robot.wheels.set_velocity(0, 0)
            imitation_state = "forward"
            turn_step = 0
        end

    elseif imitation_state == "forward" then
        local distance = seg.distance
        local segment_ticks = distance / velocity * ticks_per_second
        if segment_step < segment_ticks then
            robot.wheels.set_velocity(velocity, velocity)
            segment_step = segment_step + 1
        else
            robot.wheels.set_velocity(0, 0)
            last_angle = seg.angle
            current_segment = current_segment + 1
            segment_step = 0
            imitation_state = "turn"
        end
    end
end

function LedIsDetected(color)
    for i = 1, #robot.colored_blob_omnidirectional_camera do
        local blob = robot.colored_blob_omnidirectional_camera[i]
        local r = blob.color.red
        local g = blob.color.green
        local b = blob.color.blue
        if (color == "green") then
            if g > 200 and r < 100 and b < 100 then
                return true
            end
        elseif (color == "red") then
            if r > 200 and g < 100 and b < 100 then
                return true
            end
        elseif (color == "blue") then
            if b > 200 and r < 100 and g < 100 then
                return true
            end
        end
    end
    return false
end

function ComputeSpeedFromAngle(angle)
    dotProduct = 0.0;
    KProp = 20;
    wheelsDistance = 0.14;

    -- if the target angle is behind the robot, we just rotate, no forward motion
    if angle > math.pi / 2 or angle < -math.pi / 2 then
        dotProduct = 0.0;
    else
        -- else, we compute the projection of the forward motion vector with the desired angle
        forwardVector = {math.cos(0), math.sin(0)}
        targetVector = {math.cos(angle), math.sin(angle)}
        dotProduct = forwardVector[1] * targetVector[1] + forwardVector[2] * targetVector[2]
    end

    -- the angular velocity component is the desired angle scaled linearly
    angularVelocity = KProp * angle;
    -- the final wheel speeds are compute combining the forward and angular velocities, with different signs for the left and right wheel.
    speeds = {dotProduct * WHEEL_SPEED - angularVelocity * wheelsDistance,
              dotProduct * WHEEL_SPEED + angularVelocity * wheelsDistance}

    return speeds
end

function FollowDemonstrator()
    for i = 1, #robot.range_and_bearing do
        local rb = robot.range_and_bearing[i]
        if rb.data[1] == 99 then
            if rb.range > 100.0 then
                is_following = true
                demo_range = rb.range
                demo_angle = rb.horizontal_bearing
                break
            else
                is_following = false
                demo_range = 0
                demo_angle = 0
            end
        end
    end

    if is_following then
        local speeds = ComputeSpeedFromAngle(demo_angle)
        robot.wheels.set_velocity(speeds[1], speeds[2])
    else
        robot.wheels.set_velocity(0, 0) 
    end
end

function RandomWalk()
    local rep_force = {
        x = 0,
        y = 0
    }

    for i = 1, #robot.proximity do
        local sensor = robot.proximity[i]
        local magnitude = sensor.value
        local angle = sensor.angle
        rep_force.x = rep_force.x + magnitude * math.cos(angle)
        rep_force.y = rep_force.y + magnitude * math.sin(angle)
    end

    local rep_len = math.sqrt(rep_force.x ^ 2 + rep_force.y ^ 2)
    local rep_angle = math.atan2(rep_force.y, rep_force.x)

    if rep_len > 0.2 then
        if rep_angle > 0 then
            robot.wheels.set_velocity(turn_speed + 3, -turn_speed + 2) -- turn right
        else
            robot.wheels.set_velocity(2 - turn_speed, turn_speed + 3) -- turn left
        end
    else
        robot.wheels.set_velocity(velocity, velocity) -- go forward
    end

end

function FilterOutliers(points, radius, min_neighbors)
    local filtered = {}

    for i, pi in ipairs(points) do
        local neighbor_count = 0
        for j, pj in ipairs(points) do
            if i ~= j then
                local dx = pi.x - pj.x
                local dy = pi.y - pj.y
                local dist = math.sqrt(dx * dx + dy * dy)
                if dist < radius then
                    neighbor_count = neighbor_count + 1
                end
            end
        end

        if neighbor_count >= min_neighbors then
            table.insert(filtered, pi)
        end
    end

    return filtered
end

function AngleDistancePatternKey(pattern)
    local key = {}
    for _, seg in ipairs(pattern) do
        table.insert(key, string.format("%.1f_%.1f", seg.angle, seg.distance))
    end
    return table.concat(key, "_")
end

function FindMostRepeatedTrajectory(segments, pattern_length)
    local freq_map = {}
    local patterns = {}

    for i = 1, #segments - pattern_length + 1 do
        local pattern = {}
        for j = 0, pattern_length - 1 do
            table.insert(pattern, segments[i + j])
        end

        local key = AngleDistancePatternKey(pattern)
        if not freq_map[key] then
            freq_map[key] = 0
            patterns[key] = pattern
        end
        freq_map[key] = freq_map[key] + 1
    end

    -- Find most frequent pattern
    local best_key = nil
    local max_count = 0
    for k, v in pairs(freq_map) do
        if v > max_count then
            best_key = k
            max_count = v
        end
    end

    if best_key then
        log(string.format("[INFO] Selected pattern of length %d repeated %d times", pattern_length, max_count))
        return patterns[best_key]
    else
        return {}
    end
end

function ComputeImitationQuality(demo_segments, learner_segments)
    if #demo_segments == 0 then
        log("[quality] Error: Empty demo segments, skipping quality computation.")
        return -2
    end
    if #learner_segments == 0 then
        log("[quality] Error: Empty learner segments, skipping quality computation.")
        return 0
    end

    local len_sum = 0
    local len_diff_sum = 0
    local min_len = math.min(#demo_segments, #learner_segments)

  
    for i = 1, min_len do
        local d_seg = demo_segments[i]
        local l_seg = learner_segments[i]
        local len_diff = math.abs(d_seg.distance - l_seg.distance)
        len_sum = len_sum + d_seg.distance
        len_diff_sum = len_diff_sum + len_diff
    end

    local Ql = 1.0
    if len_sum > 0 then
        Ql = 1 - (len_diff_sum / len_sum)
    end

  
    local demo_turn_sum = 0
    local turn_diff_sum = 0

    for i = 2, min_len do
        local demo_turn = math.abs(angle_diff(demo_segments[i].angle, demo_segments[i - 1].angle))
        local learner_turn = math.abs(angle_diff(learner_segments[i].angle, learner_segments[i - 1].angle))
        local diff = math.abs(demo_turn - learner_turn)

        demo_turn_sum = demo_turn_sum + math.abs(demo_turn)
        turn_diff_sum = turn_diff_sum + diff
    end

    local Qa = 1.0
    if demo_turn_sum > 0 then
        Qa = 1 - (turn_diff_sum / demo_turn_sum)
    end

  
    
    local Qs = 1 - math.abs(#learner_segments - #demo_segments) / #demo_segments

    local L, A, S = 1, 1, 1
    local quality = (L * Ql + A * Qa + S * Qs) / (L + A + S)
    if quality < 0 then
        quality = 0
    end
    log(string.format("[quality] Ql=%.4f, Qa=%.4f, Qs=%.4f -> quality=%.4f", Ql, Qa, Qs, quality))
    return quality
end

function IsObstacleAhead(threshold)
    for i = 1, #robot.proximity do
        local sensor = robot.proximity[i]
        local angle = sensor.angle
        local value = sensor.value

        if math.abs(angle) < math.rad(45) and value > threshold then
            return true
        end
    end
    return false
end

function sort_by_nearest_neighbor(points)
    local used = {}
    local path = {}
    local current = points[1]
    table.insert(path, current)
    used[1] = true

    for _ = 2, #points do
        local nearest = nil
        local nearest_dist = math.huge
        local nearest_idx = nil
        for i, pt in ipairs(points) do
            if not used[i] then
                local d = point_distance(current, pt)
                if d < nearest_dist then
                    nearest = pt
                    nearest_dist = d
                    nearest_idx = i
                end
            end
        end
        if nearest then
            table.insert(path, nearest)
            current = nearest
            used[nearest_idx] = true
        end
    end

    return path
end

function point_distance(p1, p2)
    local dx = p1.x - p2.x
    local dy = p1.y - p2.y
    return math.sqrt(dx * dx + dy * dy)
end

function find_repeated_block(points, i, tolerance)
    local block = {i}
    local ref = points[i]

  
    local j = i - 1
    while j >= 1 and point_distance(points[j], ref) <= tolerance do
        table.insert(block, 1, j)
        j = j - 1
    end


    j = i + 1
    while j <= #points and point_distance(points[j], ref) <= tolerance do
        table.insert(block, j)
        j = j + 1
    end

    return block
end

function ComputeAngle(p1, p2)
    return math.atan2(p2.y - p1.y, p2.x - p1.x)
end

function IsSamePoint(p1, p2, epsilon)
    epsilon = epsilon or 1e-4  
    return math.abs(p1.x - p2.x) < epsilon and math.abs(p1.y - p2.y) < epsilon
end

function NormalizeAngleDifference(a1, a2)
    local diff = a2 - a1
    while diff > math.pi do diff = diff - 2 * math.pi end
    while diff < -math.pi do diff = diff + 2 * math.pi end
    return diff
end

function angle_between(p1, p2)
    local dx = p2.x - p1.x
    local dy = p2.y - p1.y
    return math.atan2(dy, dx)
end

function is_cluster_turning(cluster, angle_threshold)
   if #cluster < 3 then return false end
   for i = 2, #cluster - 1 do
      local a1 = angle_between(cluster[i - 1], cluster[i])
      local a2 = angle_between(cluster[i], cluster[i + 1])
      local delta = math.abs(a2 - a1)
      if delta > math.pi then delta = 2 * math.pi - delta end
      if delta > angle_threshold then
         return true
      end
   end
   return false
end


function ExtractMainPathUnique(points, radius, min_support, repeat_eps, min_repeat)
    local r = radius 
    local k = min_support

    if not points or #points == 0 then
        return {}
    end

    local remaining = {}
    for i, p in ipairs(points) do
        remaining[i] = { x = p.x, y = p.y }
    end

    local main_path = {}

    while #remaining > 0 do
        local ref = remaining[1]

        local cluster_idx = {}
        for i = 1, #remaining do
            if point_distance(remaining[i], ref) <= r then
                cluster_idx[#cluster_idx + 1] = i
            end
        end

        if #cluster_idx >= k then
    
            local cx, cy = 0.0, 0.0
            for _, idx in ipairs(cluster_idx) do
                cx = cx + remaining[idx].x
                cy = cy + remaining[idx].y
            end
            cx = cx / #cluster_idx
            cy = cy / #cluster_idx
            main_path[#main_path + 1] = { x = cx, y = cy }

       
            for i = #cluster_idx, 1, -1 do
                table.remove(remaining, cluster_idx[i])
            end
        else
   
            main_path[#main_path + 1] = { x = ref.x, y = ref.y }
            table.remove(remaining, 1)
        end
    end

    log(string.format("[MAIN PATH RCIA] r=%.3f, k=%d -> %d pts => %d main pts",
                      r, k, #points, #main_path))
    return main_path
end


function RemoveDuplicatePoints(points, epsilon)
    local unique_points = {}
    for i = 1, #points do
        local is_duplicate = false
        for j = 1, #unique_points do
            if IsSamePoint(points[i], unique_points[j], epsilon) then
                is_duplicate = true
                break
            end
        end
        if not is_duplicate then
            table.insert(unique_points, points[i])
        end
    end
    return unique_points
end

function remove_step_jumps(points, distance_threshold)
    local i = 1
    while i < #points do
        local d = point_distance(points[i], points[i + 1])
        if d > distance_threshold then
            table.remove(points, i)  
            i = 1                    
        else
            i = i + 1                
        end
    end
    return points
end


function region_query(points, center_idx, radius)
    local neighbors = {}
    local center = points[center_idx]
    for i, p in ipairs(points) do
        if i ~= center_idx and point_distance(center, p) <= radius then
            table.insert(neighbors, i)
        end
    end
    return neighbors
end


function find_clusters(points, radius)
    local visited = {}
    local clusters = {}

    for i = 1, #points do
        if not visited[i] then
            local cluster = {}
            local queue = {i}
            visited[i] = true

            while #queue > 0 do
                local idx = table.remove(queue, 1)
                table.insert(cluster, points[idx])
                local neighbors = region_query(points, idx, radius)
                for _, n_idx in ipairs(neighbors) do
                    if not visited[n_idx] then
                        visited[n_idx] = true
                        table.insert(queue, n_idx)
                    end
                end
            end

            table.insert(clusters, cluster)
        end
    end

    return clusters
end


function find_largest_cluster(points, radius)
    local clusters = find_clusters(points, radius)
    local largest = {}
    for _, c in ipairs(clusters) do
        if #c > #largest then
            largest = c
        end
    end
    return largest
end
