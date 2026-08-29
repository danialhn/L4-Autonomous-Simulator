import pygame
import numpy as np
import math
import random
import imageio

WIDTH = 1920
HEIGHT = 1080
VIEW_W = 1140
VIEW_H = 750
FPS = 30
TOTAL_SECONDS = 25
TOTAL_FRAMES = FPS * TOTAL_SECONDS
OUTPUT_MP4 = "autonomous_l4_zero_delay.mp4"
OUTPUT_GIF = "autonomous_l4_zero_delay.gif"

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Level 4 Autonomous // Zero-Delay Bounding Box Sensors")
clock = pygame.time.Clock()

FONT_TITLE = pygame.font.SysFont("Trebuchet MS", 18, bold=True)
FONT_BOLD = pygame.font.SysFont("Trebuchet MS", 14, bold=True)
FONT_MONO = pygame.font.SysFont("Consolas", 13)
FONT_MONO_BOLD = pygame.font.SysFont("Consolas", 13, bold=True)
FONT_TINY = pygame.font.SysFont("Consolas", 11)

FOCAL_PX = 780.0
CAM_H = 1.45
VANISH_X = VIEW_W // 2
VANISH_Y = int(VIEW_H * 0.40)
LANE_W = 3.8

BEV_CENTER_X = VIEW_W + (WIDTH - VIEW_W) // 2
BEV_ORIGIN_Y = 620
BEV_SCALE = 6.8

# Engineering Palette
COLOR_BG = (5, 8, 12)
COLOR_PANEL = (11, 15, 22)
COLOR_GRID = (25, 35, 50)
COLOR_CYAN = (0, 240, 255)
COLOR_EMERALD = (0, 255, 140)
COLOR_AMBER = (255, 180, 0)
COLOR_CRIMSON = (255, 35, 65)
COLOR_ROAD = (12, 16, 22)
COLOR_TEXT = (220, 230, 245)

class TrafficEntity:
    def __init__(self, ent_id, x_m, z_m, speed_kmh, ent_type="SEDAN", active=True):
        self.id = ent_id
        self.x = x_m
        self.z = z_m
        self.v = speed_kmh / 3.6
        self.type = ent_type
        self.active = active
        self.is_hazard = False  # Track exact hazard status natively
        if ent_type == "TRUCK":
            self.w, self.h, self.l = 2.4, 2.2, 7.2
            self.color = COLOR_AMBER
        elif ent_type == "SPORT":
            self.w, self.h, self.l = 1.95, 1.35, 4.7
            self.color = (255, 50, 120)
        elif ent_type == "SUV":
            self.w, self.h, self.l = 2.0, 1.6, 4.8
            self.color = (255, 130, 40)
        else:
            self.w, self.h, self.l = 1.9, 1.45, 4.6
            self.color = (0, 220, 210)

    def update(self, dt, ego_speed):
        if self.active:
            self.z += (self.v - ego_speed) * dt

class RoadSignEntity:
    def __init__(self, sign_id, x_m, z_m, text):
        self.id = sign_id
        self.x = x_m  
        self.z = z_m
        self.text = text

    def update(self, dt, ego_speed):
        self.z -= ego_speed * dt
        if self.z < -30.0:
            self.z += 280.0

# Fleet Initialization
fleet = [
    TrafficEntity("TRK-LEAD1", x_m=0.0, z_m=34.0, speed_kmh=66.0, ent_type="TRUCK", active=True),
    TrafficEntity("SED-FAST1", x_m=-LANE_W, z_m=-38.0, speed_kmh=125.0, ent_type="SEDAN", active=True),
    TrafficEntity("SUV-RIGHT1", x_m=LANE_W, z_m=15.0, speed_kmh=75.0, ent_type="SUV", active=True),
    TrafficEntity("SED-LEAD2", x_m=0.0, z_m=155.0, speed_kmh=70.0, ent_type="SEDAN", active=True),
    TrafficEntity("SPORT-FAST2", x_m=-LANE_W, z_m=-1000.0, speed_kmh=135.0, ent_type="SPORT", active=False),
    TrafficEntity("TRK-RIGHT2", x_m=LANE_W, z_m=110.0, speed_kmh=72.0, ent_type="TRUCK", active=True)
]

road_signs = [
    RoadSignEntity("SIGN-01", x_m=9.5, z_m=50.0, text="100"),
    RoadSignEntity("SIGN-02", x_m=-9.5, z_m=130.0, text="AUTOPILOT"),
    RoadSignEntity("SIGN-03", x_m=9.5, z_m=220.0, text="120")
]

ego_x = 0.0
ego_target_x = 0.0
ego_lane_idx = 2
ego_v = 92.0 / 3.6
ego_target_v = 95.0 / 3.6
ego_accel = 0.0

fsm_state = 0
turn_signal = "OFF"

hist_speed = [ego_v * 3.6] * 200
hist_steering = [0.0] * 200
recorded_frames = []

def project_3d_relative(rel_x, wy, wz, curvature):
    c_off = 0.5 * curvature * (wz ** 2)
    scale = FOCAL_PX / max(0.1, wz)
    sx = int(VANISH_X + (rel_x + c_off) * scale)
    sy = int(VANISH_Y + (CAM_H - wy) * scale)
    return sx, sy, scale

print("[INFO] Bootstrapping Zero-Delay Exact Clearance BSM Engine...")

frame_idx = 0
running = True

while running and frame_idx < TOTAL_FRAMES:
    dt = clock.tick(FPS) / 1000.0
    t = frame_idx / float(FPS)
    frame_idx += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

    curvature_k = 0.004 * math.sin(t * 0.25)
    road_heading = math.atan(curvature_k * 45.0)

    # 1. DYNAMIC EGO-RELATIVE SENSOR FUSION (ZERO-DELAY EXACT CLEARANCE)
    # -----------------------------------------------------------------------------
    bsm_left_active = False
    bsm_right_active = False
    bsm_left_txt = "ZONE CLEAR"
    bsm_right_txt = "ZONE CLEAR"

    ego_front = 2.4  # Front bumper coordinate
    ego_rear = -2.4  # Rear bumper coordinate

    for v in fleet:
        v.is_hazard = False
        if v.active:
            rel_x = v.x - ego_x  
            is_left = -5.5 < rel_x < -1.5
            is_right = 1.5 < rel_x < 5.5
            
            if is_left or is_right:
                v_front = v.z + v.l / 2.0
                v_rear = v.z - v.l / 2.0
                
                hazard = False
                txt = "ZONE CLEAR"
                
                # Check 1: Exact overlap alongside (turns off IMMEDIATELY when clearance > 0.5m)
                if v_front > ego_rear - 0.5 and v_rear < ego_front - 0.5:
                    hazard = True
                    txt = "ALONGSIDE [0.0s]"
                
                # Check 2: Closing Vehicle Warning (Approaching from behind up to 30m)
                elif v_front <= ego_rear - 0.5:
                    if v.v > ego_v:
                        dist = (ego_rear - 0.5) - v_front
                        ttc_rear = dist / max(0.01, (v.v - ego_v))
                        if ttc_rear < 3.0 and dist < 30.0:
                            hazard = True
                            txt = f"APPR: {dist:.1f}m | TTC: {ttc_rear:.1f}s"
                
                if hazard:
                    v.is_hazard = True
                    if is_left:
                        bsm_left_active = True
                        bsm_left_txt = txt
                    if is_right:
                        bsm_right_active = True
                        bsm_right_txt = txt

    lead_1 = fleet[0]
    lead_2 = fleet[3]
    sport_2 = fleet[4]

    active_lead = lead_1 if fsm_state < 5 else lead_2
    active_lead_gap = active_lead.z - ego_front
    ttc_lead = (active_lead_gap / max(0.1, (ego_v - active_lead.v))) if ego_v > active_lead.v else 99.0

    # 2. Sequential Finite State Machine
    if fsm_state == 0:  
        turn_signal = "OFF"
        ego_target_x = 0.0
        if active_lead_gap < 24.0:
            ego_target_v = lead_1.v
            fsm_state = 1
        else:
            ego_target_v = 95.0 / 3.6

    elif fsm_state == 1:  
        turn_signal = "LEFT"
        # Decision waits until the fast car provides 15m physical safety clearance
        if not bsm_left_active and fleet[1].z > 15.0:
            fsm_state = 2
            ego_target_x = -LANE_W
            ego_target_v = 108.0 / 3.6

    elif fsm_state == 2:  
        turn_signal = "LEFT"
        ego_target_x = -LANE_W
        if abs(ego_x - (-LANE_W)) < 0.12:
            fsm_state = 3

    elif fsm_state == 3:  
        turn_signal = "OFF"
        ego_target_x = -LANE_W
        if lead_1.z < -14.0:
            fsm_state = 4
            ego_target_x = 0.0
            ego_target_v = 95.0 / 3.6

    elif fsm_state == 4:  
        turn_signal = "RIGHT"
        ego_target_x = 0.0
        if abs(ego_x - 0.0) < 0.12:
            turn_signal = "OFF"
            fsm_state = 5

    elif fsm_state == 5:  
        turn_signal = "OFF"
        ego_target_x = 0.0
        ego_target_v = 100.0 / 3.6
        if not sport_2.active and lead_2.z < 80.0:
            sport_2.active = True
            sport_2.z = -45.0  
        if active_lead_gap < 26.0:
            fsm_state = 6

    elif fsm_state == 6:  
        turn_signal = "LEFT"
        ego_target_v = lead_2.v * 0.98  
        if not bsm_left_active and sport_2.z > 15.0:
            fsm_state = 7
            ego_target_x = -LANE_W
            ego_target_v = 110.0 / 3.6  

    elif fsm_state == 7:  
        turn_signal = "LEFT"
        ego_target_x = -LANE_W
        if abs(ego_x - (-LANE_W)) < 0.12:
            fsm_state = 8

    elif fsm_state == 8:  
        turn_signal = "OFF"
        ego_target_x = -LANE_W
        if lead_2.z < -14.0:  
            fsm_state = 9
            ego_target_x = 0.0  
            ego_target_v = 98.0 / 3.6

    elif fsm_state == 9:  
        turn_signal = "RIGHT" if abs(ego_x - 0.0) > 0.12 else "OFF"
        ego_target_x = 0.0
        if abs(ego_x - 0.0) < 0.10:
            turn_signal = "OFF"
            ego_target_v = 100.0 / 3.6

    # Continuous Lane Indexing
    if ego_x < -LANE_W * 0.5: ego_lane_idx = 1
    elif ego_x > LANE_W * 0.5: ego_lane_idx = 3
    else: ego_lane_idx = 2

    ego_x += (ego_target_x - ego_x) * 0.08
    ego_accel = 0.75 * (ego_target_v - ego_v)
    ego_accel = max(-4.8, min(2.6, ego_accel))
    ego_v += ego_accel * dt

    lat_err = (ego_x - ego_target_x) * 0.18
    steering_rad = road_heading - math.atan((0.85 * lat_err) / (ego_v + 1e-4))
    steering_deg = math.degrees(steering_rad)

    for obj in fleet: obj.update(dt, ego_v)
    for sign in road_signs: sign.update(dt, ego_v)

    hist_speed.append(ego_v * 3.6)
    hist_steering.append(steering_deg)
    if len(hist_speed) > 200: hist_speed.pop(0)
    if len(hist_steering) > 200: hist_steering.pop(0)

    # -------------------------------------------------------------
    # PANEL 1: 3D SURROUND VISION & NEURAL VECTOR SPACE (Left)
    # -------------------------------------------------------------
    view_3d = pygame.Surface((VIEW_W, VIEW_H))
    view_3d.fill(COLOR_ROAD)
    pygame.draw.rect(view_3d, (7, 10, 16), (0, 0, VIEW_W, VANISH_Y))

    scan_offset = (frame_idx * 3.5) % 12.0
    for z_g in np.arange(scan_offset, 95.0, 4.0):
        if z_g < 3.0: continue
        _, sy, _ = project_3d_relative(0, 0, z_g, curvature_k)
        if sy > VANISH_Y:
            lum = int(max(15, 120 * (1.0 - z_g / 95.0)))
            pygame.draw.line(view_3d, (lum//4, lum//2, lum//2), (0, sy), (VIEW_W, sy), 1)

    rib_pts = []
    for z_rib in np.linspace(3.0, 75.0, 32):
        lx, ly, _ = project_3d_relative(-LANE_W * 0.48, 0, z_rib, curvature_k)
        rx, ry, _ = project_3d_relative(LANE_W * 0.48, 0, z_rib, curvature_k)
        if len(rib_pts) == 0: rib_pts = [(lx, ly), (rx, ry)]
        else: rib_pts.insert(0, (lx, ly)); rib_pts.append((rx, ry))
    if len(rib_pts) >= 4:
        rib_surf = pygame.Surface((VIEW_W, VIEW_H), pygame.SRCALPHA)
        pygame.draw.polygon(rib_surf, (0, 240, 140, 55), rib_pts)
        view_3d.blit(rib_surf, (0, 0))

    for l_idx in [-1.5, -0.5, 0.5, 1.5]:
        pts = []
        lane_world_x = l_idx * LANE_W
        for z_lane in np.linspace(3.0, 95.0, 40):
            sx, sy, _ = project_3d_relative(lane_world_x - ego_x, 0, z_lane, curvature_k)
            pts.append((sx, sy))
        for i in range(len(pts) - 1):
            if pts[i][1] > VANISH_Y:
                is_outer = (abs(l_idx) == 1.5)
                th = max(1, int(4.2 * (pts[i][1] - VANISH_Y) / (VIEW_H - VANISH_Y)))
                pygame.draw.line(view_3d, COLOR_CYAN if is_outer else (190, 220, 255), pts[i], pts[i+1], th)

    for sign in road_signs:
        if 4.0 < sign.z < 100.0:
            rel_sign_x = sign.x - ego_x
            sx_base, sy_base, _ = project_3d_relative(rel_sign_x, 0.0, sign.z, curvature_k)
            sx_top, sy_top, scale = project_3d_relative(rel_sign_x, 2.6, sign.z, curvature_k)
            if sy_top > VANISH_Y:
                pygame.draw.line(view_3d, (120, 140, 160), (sx_base, sy_base), (sx_top, sy_top), max(1, int(2 * scale)))
                sw, sh = max(24, int(1.4 * scale * 12)), max(24, int(1.4 * scale * 12))
                pygame.draw.rect(view_3d, (240, 240, 240), (sx_top - sw//2, sy_top - sh//2, sw, sh), border_radius=4)
                pygame.draw.rect(view_3d, COLOR_CRIMSON, (sx_top - sw//2, sy_top - sh//2, sw, sh), width=2, border_radius=4)
                view_3d.blit(FONT_TINY.render(f"SIGN:{sign.text}", True, COLOR_CYAN), (sx_top - 20, sy_top - sh//2 - 14))

    sorted_fleet = sorted([v for v in fleet if v.active], key=lambda a: a.z, reverse=True)
    for obj in sorted_fleet:
        if obj.z < 4.0 or obj.z > 105.0: continue
        rel_obj_x = obj.x - ego_x
        hw, hh, hl = obj.w / 2.0, obj.h, obj.l / 2.0
        corners = [
            (rel_obj_x - hw, 0, obj.z - hl), (rel_obj_x + hw, 0, obj.z - hl),
            (rel_obj_x + hw, 0, obj.z + hl), (rel_obj_x - hw, 0, obj.z + hl),
            (rel_obj_x - hw, hh, obj.z - hl), (rel_obj_x + hw, hh, obj.z - hl),
            (rel_obj_x + hw, hh, obj.z + hl), (rel_obj_x - hw, hh, obj.z + hl)
        ]
        proj = [project_3d_relative(vx, vy, vz, curvature_k)[:2] for vx, vy, vz in corners]

        is_lead_target = (obj == active_lead and fsm_state in [0, 1, 5, 6])
        box_col = COLOR_CRIMSON if (is_lead_target or obj.is_hazard) else obj.color

        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for e in edges:
            p1, p2 = proj[e[0]], proj[e[1]]
            if p1[1] > VANISH_Y and p2[1] > VANISH_Y:
                pygame.draw.line(view_3d, box_col, p1, p2, 2)

        top_p = proj[7]
        tag = f"{obj.id} | {obj.z:.1f}m"
        pygame.draw.rect(view_3d, box_col, (top_p[0] - 50, max(10, top_p[1] - 22), len(tag)*7 + 8, 18), border_radius=3)
        view_3d.blit(FONT_MONO.render(tag, True, (0, 0, 0)), (top_p[0] - 46, max(12, top_p[1] - 20)))

    for obj in fleet:
        if obj.active and 4.0 < obj.z < 85.0:
            tgt_x, tgt_y, _ = project_3d_relative(obj.x - ego_x, obj.h * 0.5, obj.z, curvature_k)
            ego_sx, ego_sy, _ = project_3d_relative(0.0, 0.4, 2.0, curvature_k)
            lidar_s = pygame.Surface((VIEW_W, VIEW_H), pygame.SRCALPHA)
            pygame.draw.line(lidar_s, (0, 240, 255, 90), (ego_sx, ego_sy), (tgt_x, tgt_y), 1)
            pygame.draw.circle(lidar_s, (0, 255, 200, 240), (tgt_x, tgt_y), 4)
            view_3d.blit(lidar_s, (0, 0))

    screen.blit(view_3d, (0, 0))

    # -------------------------------------------------------------
    # PANEL 2: BEV RADAR (Right)
    # -------------------------------------------------------------
    bev_w = WIDTH - VIEW_W
    view_bev = pygame.Surface((bev_w, VIEW_H))
    view_bev.fill(COLOR_PANEL)
    bev_cx = bev_w // 2

    ego_bev_x = int(bev_cx + ego_x * BEV_SCALE)
    for r_m in [15, 30, 45, 60, 75]:
        r_px = int(r_m * BEV_SCALE)
        pygame.draw.circle(view_bev, (24, 34, 48), (ego_bev_x, BEV_ORIGIN_Y), r_px, 1)
        view_bev.blit(FONT_MONO.render(f"{r_m}m", True, (70, 100, 130)), (ego_bev_x + 8, BEV_ORIGIN_Y - r_px + 3))

    cam_fov_surf = pygame.Surface((bev_w, VIEW_H), pygame.SRCALPHA)
    pygame.draw.polygon(cam_fov_surf, (0, 240, 255, 18), [(ego_bev_x, BEV_ORIGIN_Y - 26), (ego_bev_x - 180, 0), (ego_bev_x + 180, 0)])
    
    # Left & Right BSM Dynamic Cones
    pygame.draw.polygon(cam_fov_surf, (255, 35, 65, 85) if bsm_left_active else (0, 255, 140, 25),
                        [(ego_bev_x - 14, BEV_ORIGIN_Y), (ego_bev_x - 80, BEV_ORIGIN_Y - 20), (ego_bev_x - 80, BEV_ORIGIN_Y + 150)])
    pygame.draw.polygon(cam_fov_surf, (255, 35, 65, 85) if bsm_right_active else (0, 255, 140, 25),
                        [(ego_bev_x + 14, BEV_ORIGIN_Y), (ego_bev_x + 80, BEV_ORIGIN_Y - 20), (ego_bev_x + 80, BEV_ORIGIN_Y + 150)])
    
    view_bev.blit(cam_fov_surf, (0, 0))

    for l_idx in [-1.5, -0.5, 0.5, 1.5]:
        pts_bev = []
        for z_m in np.linspace(0.0, 85.0, 30):
            bx = int(bev_cx + (l_idx * LANE_W + 0.5 * curvature_k * (z_m ** 2)) * BEV_SCALE)
            by = int(BEV_ORIGIN_Y - z_m * BEV_SCALE)
            pts_bev.append((bx, by))
        for i in range(len(pts_bev) - 1):
            is_outer = (abs(l_idx) == 1.5)
            pygame.draw.line(view_bev, COLOR_CYAN if is_outer else (45, 65, 90), pts_bev[i], pts_bev[i+1], 2 if is_outer else 1)

    pygame.draw.rect(view_bev, COLOR_CYAN, (ego_bev_x - 14, BEV_ORIGIN_Y - 26, 28, 36), border_radius=4)
    view_bev.blit(FONT_BOLD.render(f"EGO [L{ego_lane_idx}]", True, COLOR_CYAN), (ego_bev_x - 25, BEV_ORIGIN_Y + 14))

    for obj in fleet:
        if obj.active:
            abx = int(bev_cx + (obj.x + 0.5 * curvature_k * (obj.z ** 2)) * BEV_SCALE)
            aby = int(BEV_ORIGIN_Y - obj.z * BEV_SCALE)
            if 0 < abx < bev_w and 0 < aby < VIEW_H:
                bw_bev, bl_bev = max(6, int(obj.w * BEV_SCALE * 0.5)), max(6, int(obj.l * BEV_SCALE * 0.5))
                col = COLOR_CRIMSON if obj.is_hazard else obj.color
                pygame.draw.rect(view_bev, col, (abx - bw_bev, aby - bl_bev, bw_bev * 2, bl_bev * 2), border_radius=3)
                pygame.draw.line(view_bev, (255, 255, 255), (abx, aby), (abx, aby - int((obj.v - ego_v) * 2.5)), 2)
                view_bev.blit(FONT_TINY.render(f"{obj.id}", True, (220, 240, 255)), (abx + bw_bev + 4, aby - 6))

    screen.blit(view_bev, (VIEW_W, 0))

    # -------------------------------------------------------------
    # PANEL 3: PRECISION ENGINEERING HUD & TABLES (Bottom)
    # -------------------------------------------------------------
    pygame.draw.line(screen, COLOR_CYAN, (0, VIEW_H), (WIDTH, VIEW_H), 2)
    pygame.draw.rect(screen, COLOR_BG, (0, VIEW_H + 2, WIDTH, HEIGHT - VIEW_H))

    b1_x, b1_y, b1_w, b1_h = 30, VIEW_H + 20, 600, 290
    pygame.draw.rect(screen, COLOR_PANEL, (b1_x, b1_y, b1_w, b1_h), border_radius=8)
    pygame.draw.rect(screen, COLOR_GRID, (b1_x, b1_y, b1_w, b1_h), width=2, border_radius=8)
    screen.blit(FONT_TITLE.render("VEHICLE DYNAMICS TELEMETRY", True, COLOR_CYAN), (b1_x + 15, b1_y + 15))
    
    # Properly Scaled Velocity Oscilloscope
    sg_y, sg_h = b1_y + 50, 100
    pygame.draw.line(screen, COLOR_GRID, (b1_x + 15, sg_y + sg_h//2), (b1_x + b1_w - 15, sg_y + sg_h//2), 1)
    pts_v = []
    hist_len = max(1, len(hist_speed) - 1)
    for i, v in enumerate(hist_speed):
        clamp_v = max(50.0, min(130.0, v))
        gx = b1_x + 15 + int((i / hist_len) * (b1_w - 30))
        gy = sg_y + sg_h - int(((clamp_v - 50.0) / 80.0) * sg_h)
        pts_v.append((gx, gy))
    if len(pts_v) > 1: pygame.draw.lines(screen, COLOR_EMERALD, False, pts_v, 2)
    screen.blit(FONT_MONO.render(f"VELOCITY : {ego_v*3.6:6.1f} km/h", True, COLOR_EMERALD), (b1_x + 15, sg_y + 5))
    screen.blit(FONT_MONO.render(f"ACCEL    : {ego_accel:6.2f} m/s²", True, COLOR_TEXT), (b1_x + 15, sg_y + 25))

    # Properly Scaled Steering Oscilloscope
    st_y, st_h = b1_y + 170, 100
    pygame.draw.line(screen, COLOR_GRID, (b1_x + 15, st_y + st_h//2), (b1_x + b1_w - 15, st_y + st_h//2), 1)
    pts_s = []
    for i, s in enumerate(hist_steering):
        clamp_s = max(-15.0, min(15.0, s))
        gx = b1_x + 15 + int((i / hist_len) * (b1_w - 30))
        gy = st_y + st_h//2 - int((clamp_s / 15.0) * (st_h//2))
        pts_s.append((gx, gy))
    if len(pts_s) > 1: pygame.draw.lines(screen, COLOR_CYAN, False, pts_s, 2)
    screen.blit(FONT_MONO.render(f"STEERING : {steering_deg:+6.2f} deg", True, COLOR_CYAN), (b1_x + 15, st_y + 5))
    screen.blit(FONT_MONO.render(f"CURVATURE: {curvature_k*1000:+6.2f} m⁻¹", True, COLOR_TEXT), (b1_x + 15, st_y + 25))

    b2_x, b2_y, b2_w, b2_h = 650, VIEW_H + 20, 600, 290
    pygame.draw.rect(screen, COLOR_PANEL, (b2_x, b2_y, b2_w, b2_h), border_radius=8)
    pygame.draw.rect(screen, COLOR_GRID, (b2_x, b2_y, b2_w, b2_h), width=2, border_radius=8)
    screen.blit(FONT_TITLE.render("SURROUND PERCEPTION MATRIX", True, COLOR_CYAN), (b2_x + 15, b2_y + 15))
    
    th_y = b2_y + 55
    pygame.draw.line(screen, COLOR_CYAN, (b2_x + 15, th_y + 20), (b2_x + b2_w - 15, th_y + 20), 1)
    screen.blit(FONT_BOLD.render("SENSOR COMPONENT", True, COLOR_TEXT), (b2_x + 15, th_y))
    screen.blit(FONT_BOLD.render("STATE", True, COLOR_TEXT), (b2_x + 280, th_y))
    screen.blit(FONT_BOLD.render("REAL-TIME METRIC", True, COLOR_TEXT), (b2_x + 390, th_y))

    sensor_data = [
        ("Front LRR Radar (77GHz)", "LOCKED" if active_lead_gap < 30 else "SEARCH", COLOR_CRIMSON if active_lead_gap < 22 else COLOR_EMERALD, f"Dist: {active_lead_gap:5.1f}m | TTC: {ttc_lead:4.1f}s"),
        ("Left CVW Radar (24GHz)", "HAZARD" if bsm_left_active else "CLEAR", COLOR_CRIMSON if bsm_left_active else COLOR_EMERALD, bsm_left_txt),
        ("Right CVW Radar (24GHz)", "HAZARD" if bsm_right_active else "CLEAR", COLOR_CRIMSON if bsm_right_active else COLOR_EMERALD, bsm_right_txt),
        ("Solid-State LiDAR Map", "ACTIVE", COLOR_EMERALD, "Resolution: 1.2M pts/s"),
        ("V2X Comms Array", "SYNCED", COLOR_EMERALD, "Tx/Rx Latency: 1.2ms"),
        ("GNSS/RTK Positioning", "FIXED", COLOR_EMERALD, "Error: < 1.8 cm")
    ]
    
    for i, (name, state, col, metric) in enumerate(sensor_data):
        row_y = th_y + 35 + i * 30
        screen.blit(FONT_MONO.render(name, True, (180, 190, 200)), (b2_x + 15, row_y))
        screen.blit(FONT_MONO_BOLD.render(state, True, col), (b2_x + 280, row_y))
        screen.blit(FONT_MONO.render(metric, True, COLOR_TEXT), (b2_x + 390, row_y))

    b3_x, b3_y, b3_w, b3_h = 1270, VIEW_H + 20, 620, 290
    pygame.draw.rect(screen, COLOR_PANEL, (b3_x, b3_y, b3_w, b3_h), border_radius=8)
    pygame.draw.rect(screen, COLOR_GRID, (b3_x, b3_y, b3_w, b3_h), width=2, border_radius=8)
    screen.blit(FONT_TITLE.render("AUTONOMOUS DECISION KERNEL (FSM)", True, COLOR_CYAN), (b3_x + 15, b3_y + 15))

    states_dict = {
        0: "ACC_FOLLOW_LEAD1", 1: "BSM_SCAN_LEFT1", 2: "OVERTAKE_LEFT_STAGE1",
        3: "PASSING_LEAD1",    4: "RETURN_CENTER_LANE2", 5: "ACC_DECELERATE_LEAD2",
        6: "CVW_HOLD_FAST_REAR", 7: "OVERTAKE_LEFT_STAGE2", 8: "PASSING_LEAD2",
        9: "STABILIZED_CENTER_CRUISE"
    }

    ai_data = [
        ("Active Kernel State", states_dict[fsm_state], COLOR_CYAN),
        ("Current Active Lane", f"LANE {ego_lane_idx} / 3", COLOR_TEXT),
        ("Target Lane Cmd", f"LANE {2 if ego_target_x == 0 else (1 if ego_target_x < 0 else 3)}", COLOR_EMERALD),
        ("Target Velocity Cmd", f"{ego_target_v*3.6:5.1f} km/h", COLOR_TEXT),
        ("Turn Signal Relay", turn_signal, COLOR_AMBER if turn_signal != "OFF" else (100, 110, 120)),
        ("DMS Attention Focus", "100% (NEURAL SYNCED)", COLOR_EMERALD)
    ]

    for i, (key, val, col) in enumerate(ai_data):
        row_y = th_y + 10 + i * 35
        pygame.draw.rect(screen, (16, 22, 30), (b3_x + 15, row_y - 5, b3_w - 30, 28), border_radius=4)
        screen.blit(FONT_MONO.render(key, True, (180, 190, 200)), (b3_x + 25, row_y))
        screen.blit(FONT_MONO_BOLD.render(val, True, col), (b3_x + 280, row_y))

    pygame.display.flip()

    frame_arr = pygame.surfarray.array3d(screen)
    frame_arr = np.rot90(frame_arr, -1)
    frame_arr = np.fliplr(frame_arr)
    recorded_frames.append(frame_arr)

    if frame_idx % 45 == 0:
        print(f"[ENGINE] Synthesized Frame {frame_idx}/{TOTAL_FRAMES} ({(frame_idx/TOTAL_FRAMES)*100:.1f}%)")

pygame.quit()

print(f"\n[INFO] Rendering complete. Exporting Zero-Delay CVW outputs...")
imageio.mimsave(OUTPUT_MP4, recorded_frames, fps=FPS, macro_block_size=1)
print(f"✅ [SUCCESS] Video Exported: {OUTPUT_MP4}")
imageio.mimsave(OUTPUT_GIF, recorded_frames[::2], fps=FPS // 2)
print(f"✅ [SUCCESS] GIF Exported: {OUTPUT_GIF}")