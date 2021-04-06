import numpy as np
from math import *
import sys 
import matplotlib.pyplot as plt
import pygame

class particle:
    def __init__(self, R , r, a, angle, center, mass):
        self.R = R         # radius of large circle
        self.r = r         # radius of small circle (R >= r)
        self.a = a         # width/2 of a particle
        self.angle = angle # anti-clockwise
        self.center = center
        self.m = mass
    
    def left_center(self):
        LC = np.zeros(2)
        LC[0] = self.center[0] - ((self.a - self.r)*cos(self.angle))/2
        LC[1] = self.center[1] - ((self.a - self.r)*sin(self.angle))/2
        return LC

    def right_center(self):
        RC = np.zeros(2)
        RC[0] = self.center[0] + ((self.a - self.r)*cos(self.angle))/2
        RC[1] = self.center[1] + ((self.a - self.r)*sin(self.angle))/2
        return RC

    def find_min_of3(self, vec):
        if vec[0] < vec[1]:
            if vec[0] < vec[2]:
                return vec[0]
            else:
                return vec[2]
        else:
            if vec[1] < vec[2]:
                return vec[1]
            else:
                return vec[2]

    def equal(self, vec1, vec2):
        if len(vec1) == len(vec2):
            for i in range(len(vec1)):
                if vec1[i] != vec2[i]:
                    return False
                    break
            return True
        else:
            sys.exit("can't compair")

    def distance_twopoint(self, point1, point2):
        vec = point1 - point2
        return np.linalg.norm(vec)

    def get_distance(self, point):
        dist = np.zeros(3)
        dist[0] = self.distance_twopoint(self.left_center(), point) - self.r
        dist[1] = self.distance_twopoint(self.center, point) - self.R
        dist[2] = self.distance_twopoint(self.right_center(), point) - self.r
        return self.find_min_of3(dist)

    def direction_twopoint(self, point1, point2): # a unit vector point to point2
        vec = point2 - point1
        vec_len = np.linalg.norm(vec)
        return vec / vec_len
    
    def get_direction(self, point):
        dist_left   = self.distance_twopoint(self.left_center(), point) - self.r
        dist_center = self.distance_twopoint(self.center, point) - self.R
        dist_right  = self.distance_twopoint(self.right_center(), point) - self.r
        dist        = np.array([dist_left, dist_center, dist_right])
        min_dist    = self.find_min_of3(dist)
        if min_dist == dist_left:
            direction = self.direction_twopoint(point, self.left_center()) # point to left-center
        elif min_dist == dist_center:
            direction = self.direction_twopoint(point, self.center) # point to center
        else:
            direction = self.direction_twopoint(point, self.right_center()) # point to right-center
        return direction

    def dist_point_line(self, point, p1, p2): # a point and a line (p1, p2)
        line_vec       = p2 - p1
        pnt_vec        = point - p1
        line_len       = np.linalg.norm(line_vec)
        line_unitvec   = line_vec / line_len
        pnt_vec_scaled = pnt_vec / line_len
        temp           = line_unitvec.dot(pnt_vec_scaled)
        if temp < 0.0:
            temp = 0.0
        elif temp > 1.0:
            temp = 1.0
        nearest       = line_vec * temp
        dist          = pnt_vec - nearest
        nearest       = nearest + p1
        distance      = np.linalg.norm(dist)
        #n = dist / distance
        #t = np.array([-n[1], n[0]])
        return distance, nearest # distance and the nearest point

    def particle_line(self, p1, p2): # point1 and point2 forms a line
        line_lc ,nearest_lc    = self.dist_point_line(self.left_center(), p1, p2)
        line_cent ,nearest_cen = self.dist_point_line(self.center, p1, p2)
        line_rc ,nearest_rc    = self.dist_point_line(self.right_center(), p1, p2)
        if line_cent <= 1e-3 or line_lc <= 1e-3 or line_rc <= 1e-3:
            sys.exit("line is zero")
        mindist = self.find_min_of3([line_lc, line_cent, line_rc]) # the distance between a particle and a line
        if mindist == line_lc:
            forcepoint = self.left_center() # the force-point for a particle when interacting with a line
            direction  = self.left_center() - nearest_lc # point to the force-point
            radii = self.r
            #mindist -= radii
        elif mindist == line_cent:
            forcepoint = self.center
            direction  = self.center - nearest_cen
            radii = self.R
            #mindist -= radii
        else:
            forcepoint = self.right_center()
            direction  = self.right_center() - nearest_rc
            radii = self.r
            #mindist -= radii

        if mindist <= 1e-3:
            sys.exit("mindist is zero")
        #return forcepoint, mindist, direction
        return forcepoint, mindist, direction, radii

    def particle_particle(self, mc): # mc is another particle
        if isinstance(mc, particle):
            p1 = mc.left_center()
            p2 = mc.right_center()
            line_lc ,nearest_lc    = self.dist_point_line(self.left_center(), p1, p2)
            line_cent ,nearest_cen = self.dist_point_line(self.center, p1, p2)
            line_rc ,nearest_rc    = self.dist_point_line(self.right_center(), p1, p2)
            '''if line_cent <= 1e-3 or line_lc <= 1e-3 or line_rc <= 1e-3:
                sys.exit("line is zero")'''   # 这里的line是0
            mindist                = self.find_min_of3([line_lc, line_cent, line_rc]) # the distance between a particle and a line
            forcepoint             = np.zeros(2)
            mc_lc_fp               = self.distance_twopoint(forcepoint, mc.left_center())
            mc_cen_fp              = self.distance_twopoint(forcepoint, mc.center)
            mc_rc_fp               = self.distance_twopoint(forcepoint, mc.right_center())
            min_mc                 = self.find_min_of3([mc_lc_fp, mc_cen_fp, mc_rc_fp])
            if mindist == line_lc: # left-center is the force-point
                forcepoint = self.left_center() # the force-point for a particle when interacting with a line
                direction  = self.left_center() - nearest_lc # point to the force-point (normal vector)
                if min_mc == mc_lc_fp or min_mc == mc_rc_fp:
                    radii = 2 * self.r
                    #mindist -= radii
                else:
                    radii = (self.r + self.R)
                    #mindist -= radii
            elif mindist == line_cent:
                forcepoint = self.center
                direction  = self.center - nearest_cen
                if min_mc == mc_lc_fp or min_mc == mc_rc_fp:
                    radii = (self.r + self.R)
                    #mindist -= radii
                else:
                    radii = 2 * self.R
                    #mindist -= radii
            else:
                forcepoint = self.right_center()
                direction  = self.right_center() - nearest_rc
                if min_mc == mc_lc_fp or min_mc == mc_rc_fp:
                    radii = 2 * self.r
                    #mindist -= radii
                else:
                    radii = (self.r + self.R)
                    #mindist -= radii
        else:
            sys.exit("mc is not a particle")

        '''if mindist <= 1e-3:
            sys.exit("mindist is zero")'''

        return forcepoint, mindist, direction, radii
        #return forcepoint, mindist, direction

    def hit_mc(self, mc): # mc is another particle
        if isinstance(mc, particle):
            mindist = self.particle_particle(mc)[1]
            radii   = self.particle_particle(mc)[3]
            if mindist <= radii:
                return True
            else:
                return False
        else:
            sys.exit("mc is not a particle")

    def hit_line(self, p1, p2):
        mindist = self.particle_line(p1, p2)[1]
        radii   = self.particle_line(p1, p2)[3]
        if mindist <= radii:
            return True
        else:
            return False
  
    def get_area(self):
        return pi * self.a * self.R

    def get_moment_of_inertia(self):
        J_large = 0.5 * self.m * self.R ** 2
        J_small = self.m * (0.5 * self.r ** 2 + (self.a - self.r) ** 2)
        return J_large + 2 * J_small
    
    def get_bounds(self): # 外切矩形 (left, top, width, height)
        if abs(self.angle % pi) < 1.0e-10:
            halfWidth = self.a
            halfHeight = self.R
        elif (abs(self.angle % pi) - pi/2) < 1.0e-10:
            halfWidth = self.R
            halfHeight = self.a
        else:
            k = (-1)*tan(self.angle)
            halfHeight = sqrt(((self.a**2)*k*k + (self.R**2))/(k*k+1))
            halfWidth = sqrt(self.a**2 + self.R**2 -halfHeight**2)
        x = self.center[0] - halfWidth
        y = self.center[1] + halfHeight

        return x, y, 2*halfWidth, 2*halfHeight


class Room:
    def __init__(self, room_size, alpha, A, w, phi, totaltime, h):
        self.room_size = room_size # length
        self.A = A
        self.w = w
        self.phi = phi
        self.tot = totaltime
        self.h = h
        self.alpha = alpha
        self.num_walls = 4
        self.walls = np.array([[[1, 1], [1, self.room_size]],                                               # left wall
                        [[1, self.room_size], [self.room_size, self.room_size * (1 - tan(self.alpha))]],    # bottom wall
                        [[self.room_size, self.room_size * (1 - tan(self.alpha))], [self.room_size, 1]],    # right wall
                        [[self.room_size, 1], [1, 1]]])                                                     # top wall
        self.spawn_zone = np.array([[10, self.room_size -10], [self.room_size - 100, self.room_size - 10]])  
    
    def vib(self, t):
        return self.A * sin(self.w * t - self.phi)

    def vibration(self):
        vibration = np.zeros(self.tot)
        time = np.arange(1, self.tot+1) * self.h
        for i in range(self.tot):
            vibration[i] = self.vib(time[i])
        return vibration
    
    def vib1(self, t):
        return self.A * self.w * cos(self.w * t - self.phi)

    def vibration1(self):
        vibration1 = np.zeros(self.tot)
        time = np.arange(1, self.tot+1) * self.h
        for i in range(self.tot):
            vibration1[i] = self.vib1(time[i])
        return vibration1
    
    def vib2(self, t):
        return (- (self.w ** 2)) * self.vib(t)

    def vibration2(self):
        vibration2 = np.zeros(self.tot)
        time = np.arange(1, self.tot+1) * self.h
        for i in range(self.tot):
            vibration2[i] = self.vib2(time[i])
        return vibration2

    def walls_vib(self):
        time = np.arange(1, self.tot+1) * self.h
        wall_vib = np.zeros((4,2,2,self.tot))
        zero = np.zeros(self.tot)
        for i in range(self.tot):
            if i <= self.tot / 8:
                wall_vib[:,:,:,i] = self.walls * (1 + zero[i])
            else:
                wall_vib[:,:,:,i] = self.walls * (1 + self.vib(time[i]))
        return wall_vib
                                         
    def get_wall(self, n, k):              # gives back the endpoints of the nth wall
        return self.walls_vib()[n,:,:, k]

    def get_num_walls(self):            # gives back the number of walls
        return self.num_walls

    def get_spawn_zone(self):            # gives back the spawn_zone
        return self.spawn_zone

    def get_room_size(self):            # gives back the size of the room
        return self.room_size


#self.N, self.L, self.tau, self.room, self.alpha, self.num_steps, self.a, self.R, self.r, self.center, self.angle, self.m
class Equations:
    def __init__(self, num_particles, h, room, alpha, totaltime, a, R, r, mass):
        self.room = room                            #import class room as room
        self.N = num_particles                      
        self.m = mass        # (1,N)                          
        self.r = r           # (1,N)
        self.R = R           # (1,N)
        self.a = a
        #self.y = y
        #self.center = center # (2,N,tot) 
        #self.angle = angle   # (1,N,tot)                        
        self.h = h                              # time-step (s)
        self.K = 10000
        self.miu_w = 0.7
        self.miu_p = 0.1
        self.g = 9.81
        self.tot = totaltime
        # wall 
        self.alpha = alpha
        self.numwalls = self.room.get_num_walls()   # number of walls 
        self.walls = self.room.walls_vib()          
        self.vib = self.room.vibration()
        self.vib1 = self.room.vibration1()
        self.vib2 = self.room.vibration2()

    # R , r, a, angle, center, mass
    def part(self, i, y, angle, k): # particle i at time k
        return particle(self.R[i], self.r[i], self.a[i], angle[:,i,k], y[:,i,k], self.m[i])      
        
    def wall_distance(self, i, j, y, angle, k): # particle i ,wall j ,time k ,point to i (forcepoint)
        temp_wall = self.walls[j,:,:,k]
        p1        = temp_wall[0,:]
        p2        = temp_wall[1,:]
        forcepoint, mindist, direction, radii = self.part(i, y, angle, k).particle_line(p1, p2)
        distance  = mindist
        dist      = np.linalg.norm(direction)
        n         = direction / dist  # normal vector ,center to wall
        t         = np.array([-n[1], n[0]]) # clockwise
        
        return forcepoint, distance, n, t, radii

    def particle_distance(self, i, j, y , angle, k): # point to i
        particle_i = self.part(i, y, angle, k)
        particle_j = self.part(j, y, angle, k)
        forcepoint, mindist, direction, radii = particle_i.particle_particle(particle_j)
        distance   = mindist
        dist       = np.linalg.norm(direction)
        n          = direction / dist  
        t          = np.array([-n[1], n[0]])

        return forcepoint, distance, n, t, radii

    def f_ij(self, i, j, y, angle, k):
        particle_i = self.part(i, y, angle, k)
        particle_j = self.part(j, y, angle, k)
        #rad_ij = particle_i.particle_particle(particle_j)[3]
        dist, n, t, radii = self.particle_distance(i, j, y, angle, k)[1:]
        if particle_i.hit_mc(particle_j):
            a = self.K * abs(dist - radii)
            b = self.miu_p * a
            return a * n + b * t
        else:
            return np.zeros(2)
    
    def n(self, i, k): # time k
        if self.vib2[k] >= 9.81:
            return 0
        else:
            return self.m[i] * cos(self.alpha) * (self.g - self.vib2[k])

    def fr1(self, i, k): # time k
        return self.m[i] * sin(self.alpha) * (self.g - self.vib2[k])

    def fr2(self, i, k): # time k
        return -self.miu_w * self.n(i, k) * np.sign(-self.fr1(i, k))
    
    def f_iW(self, i, j, y, angle, k): 
        dist, per, t, radii = self.wall_distance(i, j, y, angle, k)[1:]
        fr = np.zeros((self.N, self.tot))
        #a = self.K * self.d(self.radii[i] - dist) + self.n(i, k)        
        temp_wall = self.walls[j,:,:,k]
        p1        = temp_wall[0,:]
        p2        = temp_wall[1,:]
        particle_i = self.part(i, y, angle, k)
        if particle_i.hit_line(p1, p2) == False or self.vib2[k] >= 9.81:
            if particle_i.hit_line(p1, p2) == True:
                if abs(self.fr1(i, k)) >= self.miu_w * self.n(i, k):
                    fr[i,k] = self.fr2(i, k)
                else:
                    fr[i,k] = self.fr1(i, k)
            a = self.K * abs(dist - radii) + self.n(i, k) 
            if particle_i.hit_line(p1, p2) == False:
                fr[i,k] = 0
                a = 0
        else:
            a = self.K * abs(dist - radii) + self.n(i, k) 
            if abs(self.fr1(i, k)) >= self.miu_w * self.n(i, k):
                fr[i,k] = self.fr2(i, k)
            else:
                fr[i,k] = self.fr1(i, k)

        return a * per + fr[i,k] * t ### check the direction of fr

    def equal(self, vec1, vec2):
        if len(vec1) == len(vec2):
            for i in range(len(vec1)):
                if vec1[i] != vec2[i]:
                    return False
                    break
            return True
        else:
            sys.exit("can't compair")
    
    def moment_ij(self, i, j, y, angle, k):
        p_i        = self.part(i, y, angle, k)
        forcepoint = self.particle_distance(i, j, y, angle, k)[0]
        cen_point  = p_i.center
        LC_point   = p_i.left_center()
        RC_point   = p_i.right_center()
        if self.equal(forcepoint, cen_point):
        #if forcepoint.all() == cen_point.all():
            moment = 0
        else:
            if self.equal(forcepoint, LC_point):
            #if forcepoint.all() == LC_point.all():
                vec_p_i = cen_point - LC_point
            else:
                vec_p_i = RC_point - cen_point
            fij = self.f_ij(i, j, y, angle, k)
            moment = np.cross(vec_p_i, fij)
        
        return moment

    def moment_gravity(self, i, y, angle, k):
        p_i = self.part(i, y, angle, k)
        vec = np.array([0, 1])
        mg_i = self.g * vec
        L = self.room.room_size
        p1 = np.array([0, L])
        p2 = np.array([L, L])
        forcepoint = p_i.particle_line(p1, p2)[0]
        cen_point  = p_i.center
        LC_point   = p_i.left_center()
        RC_point   = p_i.right_center()
        if self.equal(forcepoint, cen_point):
            moment = 0
        else:
            if self.equal(forcepoint, LC_point):
                vec_p_i = cen_point - LC_point
            else:
                vec_p_i = RC_point - cen_point
            moment = np.cross(vec_p_i, mg_i)

        return moment

    def moment_iw(self, i, j, y, angle, k):
        p_i        = self.part(i, y, angle, k)
        forcepoint = self.wall_distance(i, j, y, angle, k)[0]
        cen_point  = p_i.center
        LC_point   = p_i.left_center()
        RC_point   = p_i.right_center()
        vec_p_i = np.zeros(2)
        if self.equal(forcepoint, cen_point):
            moment = 0
        else:
            if self.equal(forcepoint, LC_point):
                vec_p_i = cen_point - LC_point            
            else: 
                vec_p_i = RC_point - cen_point
            f_iwj = self.f_iW(i, j, y, angle, k)
            moment = np.cross(vec_p_i, f_iwj)
        #else:
            #sys.exit("wrong")
        
        return moment, vec_p_i

    def vec_w(self, y, angle, k):
        vec_wall = np.zeros((2, self.N))
        for i in range(self.N):
            for j in range(self.numwalls):
                vec = self.moment_iw(i, j, y, angle, k)[1]
                vec_wall[:, i] += vec
        return vec_wall

    #The interacting force of the particles to each other  
    def f_particles(self, y, angle, k):
        f_particles = np.zeros((2, self.N))
        fij     = np.zeros(((2, self.N, self.N)))
        for i in range(self.N-1):
            for j in range(self.N-1-i):
                    fij[:,i,j+i+1] = self.f_ij(i, j+i+1, y, angle, k)
                    fij[:,j+i+1,i] = -fij[:,i,j+i+1]
        f_particles = np.sum(fij, 2) # total force for each particle
        return f_particles

    #The force of each wall acting on each particles
    def f_wp(self, y, angle, k):
        f_wall = np.zeros((2, self.N))
        for i in range(self.N):
            for j in range(self.numwalls):
                f_wall[:, i] += self.f_iW(i, j, y, angle, k)
        return f_wall

    def m_particles(self, y, angle, k):
        m_particles = np.zeros((1, self.N))
        mij = np.zeros(((1, self.N, self.N)))
        for i in range(self.N-1):
            for j in range(self.N-1-i):
                m_ij = self.moment_ij(i, j+i+1, y, angle, k)
                mij[:,i,j+i+1] = m_ij
                mij[:,j+i+1,i] = -mij[:,i,j+i+1]
        m_particles = np.sum(mij, 2) # total force for each pedestrian
        return m_particles

    def m_wp(self, y, angle, k):
        m_wall = np.zeros((1, self.N))
        for i in range(self.N):
            for j in range(self.numwalls):
                m_w = self.moment_iw(i, j, y, angle, k)[0]
                m_wall[:, i] += m_w
        return m_wall
    
    def m_gra(self, y, angle, k):
        m_gra = np.zeros((1, self.N))
        for i in range(self.N):
            m_g = self.moment_gravity(i, y, angle, k)
            m_gra[:, i] = m_g
        return m_gra


    #Calculates the accelaration of each particle
    def f(self, y, angle, k):########
        a = np.zeros((2, self.N))
        a[1,:] = 1
        acc = self.f_particles(y, angle, k) / self.m + self.f_wp(y, angle, k) / self.m + self.g * a
        return acc

    def moment(self, y, angle, k): # 角加速度
        inertia = np.zeros(self.N)
        for i in range(self.N):
            p_i = self.part(i, y, angle, k)
            J_i = p_i.get_moment_of_inertia()
            inertia[i] = J_i
        beta = self.m_particles(y, angle, k) / inertia + self.m_wp(y, angle, k) / inertia + self.m_gra(y, angle, k) / inertia
        return beta


def exp_euler(y0, v0, angle0, omega0, f, moment, N_steps, dt):

    y = np.zeros((y0.shape[0], y0.shape[1], N_steps))
    v = np.zeros((y0.shape[0], y0.shape[1], N_steps))
    a = np.zeros((y0.shape[0], y0.shape[1], N_steps))

    angle = np.zeros((angle0.shape[0], angle0.shape[1], N_steps))
    omega = np.zeros((angle0.shape[0], angle0.shape[1], N_steps))
    beta  = np.zeros((angle0.shape[0], angle0.shape[1], N_steps))

    y[:,:,0] = y0
    v[:,:,0] = v0

    angle[:,:,0] = angle0
    omega[:,:,0] = omega0


    for k in range(N_steps-1):
        #print(100*k/N_steps, '% done.')
        a[:,:,k]   = f(y[:,:,:], angle[:,:,:], k)
        v[:,:,k+1] = v[:,:,k] + dt*a[:,:,k]
        y[:,:,k+1] = y[:,:,k] + dt*v[:,:,k+1]

        beta[:,:,k]    = moment(y[:,:,:], angle[:,:,:], k)
        omega[:,:,k+1] = omega[:,:,k] + dt * beta[:,:,k]
        angle[:,:,k+1] = angle[:,:,k] + dt * omega[:,:,k+1]
    #print(angle)
     
    return y, angle, beta


# self.y, self.room, wait_time, self.a, self.R, self.r, self.angle, sim_size
def display_events(movement_data, room, wait_time, a, R, r, m, angle, sim_size):

    # colors
    background_color = (170, 170, 170)            # grey
    particle_color1 = (250, 0, 0)                 # red
    particle_color2 = (0, 250, 0)                 # green
    particle_color3 = (0, 0, 250)                 # blue
    particle_color4 = (128, 0, 128)               # purple
    object_color    = (0, 0, 0)                      # black

    # variable for initializing pygame
    normalizer = int(sim_size/room.get_room_size())     # the ratio (size of image) / (size of actual room) 
    map_size = (room.get_room_size()*normalizer + 100,  #size of the map
                room.get_room_size()*normalizer + 100)  #plus a little free space
    wait_time = wait_time                               #time that the simultation waits between each timestep
    wait_time_after_sim = 1000  # 1s                    #waittime after simulation
    movement_data_dim = movement_data.shape         
    num_particles = movement_data_dim[1]          #number of indiciduals in the simulation
    num_time_iterations = movement_data_dim[2]  #number of timesteps
    num_walls = room.get_num_walls()            #number of walls

    pygame.init()                                 #initialize the intanz
    simulate=False                                #variable to indicate if the simulation is running
    font = pygame.font.Font(None, 32)             #create a new object of type Font(filename, size)
    worldmap = pygame.display.set_mode(map_size)
    
    while True:
        # start simulation if any key is pressed and quits pygame if told so
        for event in pygame.event.get(): 
            if event.type == pygame.KEYDOWN:
                simulate=True
                if event.key == pygame.K_0:
                    pygame.quit()
            elif event.type == pygame.QUIT:
                pygame.quit()
        worldmap.fill(0) # black
        #This creates a new surface with text already drawn onto it
        text = font.render('Press any key to start , Press 0 to quit', True, (255, 255, 255))
        #printing the text starting with a 'distance' of (100,100) from top left
        worldmap.blit(text, (10, 100))
        pygame.display.update()
        
        if simulate == True:
            # print the map for each timestep
            for t in range(num_time_iterations): # number of time step
                # quit the simulation if told so
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                
                #initialize the map with background color
                worldmap.fill(background_color)
        
                '''#draw each particle for timestep t
                for particle in range(num_particles):
                    pygame.draw.circle(worldmap, particle_color, 
                                    ((normalizer*movement_data[0, particle, t] + 50).astype(int),
                                    (normalizer*movement_data[1, particle, t] + 50).astype(int)),
                                    int(normalizer * radii[particle]), 0)'''
                
                #draw each particle for timestep t
                for i in range(num_particles):
                    particle_i = particle(R[i], r[i], a[i], angle[:,i,t], movement_data[:,i,t], m[i])
                    C_i = movement_data[:,i,t]
                    LC_i = particle_i.left_center()
                    RC_i = particle_i.right_center()
                    pygame.draw.circle(worldmap, particle_color2, 
                                    ((normalizer * C_i[0] + 50).astype(int),
                                    (normalizer * C_i[1] + 50).astype(int)),
                                    int(normalizer * R[i]), 0)
                    pygame.draw.circle(worldmap, particle_color1, 
                                    ((normalizer * LC_i[0] + 50).astype(int),
                                    (normalizer * LC_i[1] + 50).astype(int)),
                                    int(normalizer * r[i]), 0)
                    pygame.draw.circle(worldmap, particle_color3, 
                                    ((normalizer * RC_i[0] + 50).astype(int),
                                    (normalizer * RC_i[1] + 50).astype(int)),
                                    int(normalizer * r[i]), 0)
                    
                    pygame.draw.lines(worldmap, particle_color4, True, 
                                    (normalizer * LC_i + 50, normalizer * RC_i + 50), 2)
        
                #draw each object for timestep t
                for wall in range(num_walls):
                    pygame.draw.lines(worldmap, object_color, True, 
                                    normalizer * room.get_wall(wall, t) + 50, 2)

                #update the map
                pygame.display.update()
                #wait for a while before drawing new positions
                pygame.time.wait(wait_time)
            simulate=False
            text = font.render('SIMULATION FINISHED', True, (0, 255, 0))
            worldmap.blit(text, (100, 90))
            pygame.display.update()
            pygame.time.wait(wait_time_after_sim)


class Simulation:
    def __init__(self, num_particles, num_steps, alpha, A, a, R, r, w, phi, room_size, method = "exp_euler", tau = 0.02):

        self.L = room_size                  # size of square room (m)
        self.N = num_particles              # quantity of particles
        self.tau = tau                      # time-step (s)
        self.num_steps = num_steps          # number of steps for integration
        # viberation
        self.A = A
        self.w = w
        self.alpha = alpha
        self.phi = phi
        # Particle information
        #self.radii = 5 * np.ones(self.N)                # radii of particles (m)
        self.m = 80 * np.ones(self.N)                   # mass of particles (kg)
        self.v_0 = np.zeros(self.N)                     # desired velocity (m/s)
        self.forces = None                              # forces on the particles
        self.v = np.zeros((2, self.N, self.num_steps))  # Three dimensional array of velocity
        self.y = np.zeros((2, self.N, self.num_steps))  # Three dimensional array of place: x = coordinates, y = particle, z=Time
        self.a = a * np.ones(self.N)
        self.R = R * np.ones(self.N)
        self.r = r * np.ones(self.N)
        #self.center = self.y[:,:,0]
        #self.center = self.y
        self.angle = np.zeros((1, self.N, self.num_steps))
        #self.particle = particle()
        self.omega = np.zeros((1,self.N, self.num_steps))
        self.beta = np.zeros((1, self.N, self.num_steps))
        self.omega_0 = 0.0 * np.ones(self.N)
        # other
        
        self.room = Room(self.L, self.alpha, self.A, self.w, self.phi, self.num_steps, tau)  # kind of room the simulation runs in
        self.method = exp_euler  # method used for integration
        self.equ = Equations(self.N, self.tau, self.room, self.alpha, self.num_steps, 
                            self.a, self.R, self.r, self.m)  # initialize Differential equation
 
    # function set_time, set_steps give the possiblity to late change these variable when needed
    def set_steps(self, steps):
        self.num_steps = steps

    # function to change the methode of integration if needed
    def set_methode(self, method):
        self.method = exp_euler

    # yields false if particles don't touch each other and true if they do
    def dont_touch(self, i, pos): 
        for j in range(i - 1):
            if np.linalg.norm(np.array(pos) - self.y[:, j, 0]) < 4 * self.R[i]:
                return True
        return False

    # fills the spawn zone with particles with random positions
    def fill_room(self):
        spawn = self.room.get_spawn_zone()
        len_width = spawn[0, 1] - spawn[0, 0]
        len_height = spawn[1, 1] - spawn[1, 0]
        max_len = max(len_height, len_width)

        # checks if the area is too small for the particles to fit in
        area_particle = 0
        for i in range(self.N):
            particle_i = particle(self.R[i], self.r[i], self.a[i], self.angle[:,i,0], self.y[:,i,0], self.m[i])
            area_i = particle_i.get_area()
            area_particle += area_i
        if area_particle >= 0.7 * max_len ** 2:
            sys.exit('Too much particles! ')
        # checks if the particle touches another particle/wall and if so gives it a new random position in the spawn-zone 
        for i in range(self.N):
            # The particle don't touch the wall
            x = len_width * np.random.rand() + spawn[0, 0]
            y = len_height * np.random.rand() + spawn[1, 0]
            pos = [x, y]

            # The particles don't touch each other

            while self.dont_touch(i, pos):#touch is true, change x,y coordinate
                for i in range(10):
                    x = len_width * np.random.rand() + spawn[0, 0]
                    y = len_height * np.random.rand() + spawn[1, 0]
                    pos = [x, y]
                
            self.y[:, i, 0] = pos
            self.angle[:, i, 0] = 3.14 * np.random.rand()


        self.v[:, :, 0] = self.v_0
        self.omega[:, :, 0] = self.omega_0

    # calls the method of integration with the starting positions, diffequatial equation, number of steps, and delta t = tau
    def run(self):
        self.y, self.angle, self.beta = self.method(self.y[:, :, 0], self.v[:, :, 0], \
            self.angle[:,:,0], self.omega[:,:,0],self.equ.f, self.equ.moment,self.num_steps, self.tau)

    # Displays the simulation in pygame
    def show(self, wait_time, sim_size):
        

        display_events(self.y, self.room, wait_time, self.a, self.R, self.r, self.m, self.angle, sim_size)

sim = Simulation(num_particles=10, num_steps=500, A = 0.005, alpha = 0.01 ,w = 40,phi = 10, a = 24, R = 10, r = 8, method="exp_euler", room_size=400)
sim.fill_room()                 # fills the spawn zone with random particle
sim.run()                       
sim.show(wait_time=50, sim_size=600)   