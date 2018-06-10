#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
MPC trajectory planner

Max Ahlberg
"""


import numpy as np
import scipy
import matplotlib.pyplot as plt
import copy
import math
import cubic_spline_planner
import cvxpy
from cvxpy import *

# Parameter
ITERATION = 0
N = 20  # MPC Horizon
DT = 6  # delta s in [m]
dt = 0.1 # delta t in [s]

MAX_ROAD_WIDTH = 3.0  # maximum road width [m]
MAX_PSI = math.pi / 2 #max heading error 45 degrees
MAX_SPEED = 250.0 / 3.6  # maximum speed [m/s]
MAX_CURVATURE = 1.0  # maximum curvature [1/m]
MAX_ACCEL = 10.0  # maximum acceleration [m/ss]

MAX_C = 0.02 # maximum change in curvature [1/m^2]
MAX_J = 100.0 # maximum change in acceleration [m/s^3]


#Vehicle parameters
L = 3.0  # [m] wheel base of vehicle
lr = L*0.5 #[m]
lf = L*0.5 #[m]
Width = 2.0  # [m] Width of the vehicle

show_animation = True


class quinic_polynomial: #Skapar ett 5e grads polynom som beräknar position, velocity och acceleration

    def __init__(self, xs, vxs, axs, xe, vxe, axe, T): # (position_xs, Velocity_xs, Acceleration_xs, P_xe, V_xe, A_xe, Time )

        # calc coefficient of quinic polynomial
        self.xs = xs 
        self.vxs = vxs
        self.axs = axs
        self.xe = xe
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0 #Varför accelerationen delat med 2? -För att de skall bli rätt dimensioner i slutandan

        A = np.array([[T**3, T**4, T**5],
                      [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                      [6 * T, 12 * T ** 2, 20 * T ** 3]])
        b = np.array([xe - self.a0 - self.a1 * T - self.a2 * T**2,
                      vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b) #Antagligen matris invers som löser a3,a4,a5. En form av jerk, jerk_derivata, jerk dubbelderivata

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]
    def calc_point(self, t): # point on xs at t
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + \
            self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5

        return xt

    def calc_first_derivative(self, t): #speed in point at t
        xt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t**2 + 4 * self.a4 * t**3 + 5 * self.a5 * t**4

        return xt

    def calc_second_derivative(self, t): # acceleration in point at t
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + 20 * self.a5 * t**3

        return xt


class quartic_polynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, T):

        # calc coefficient of quinic polynomial
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * T ** 2, 4 * T ** 3],
                      [6 * T, 12 * T ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + \
            self.a3 * t**3 + self.a4 * t**4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t**2 + 4 * self.a4 * t**3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2

        return xt


class MPC_path:

    def __init__(self):
        self.t = []
        #States
        self.d = [] #Deviation from centerline
        self.psi = [] #Yaw angle of car
        self.v = [] #Speed
        self.K = [] #Curvature of car trajectory
        self.a = [] #Acceleration
        #from cost function
        self.scar = []
        #Inputs
        self.C = []
        self.J = []
        #Track
        self.s = [] #Position along the S-coordinate in frenet frame
        self.x = [] #Global
        self.y = [] #Global
        self.yaw = [] #Yaw angle of the Track
        self.c = []   #Curvature of the Track
        self.ds = []

        #Second model states
        self.S = [] #Position along centerline
        self.D = [] #Deviation from centerline
        self.S_d = [] #Speed along centerline
        self.D_d = [] #speed perp. from centerline

        #inputs
        self.S_dd = [] #Acceleration along centerlin
        self.D_dd = [] #Acceleration perp. centerline


class Vehcile_state:

    def __init__(self):
        self.t = []
        #States
        self.d = [] #Deviation from centerline
        self.psi = [] #Yaw angle of car
        self.v = [] #Speed
        self.K = [] #Curvature of car car (trajectory-curvature)
        self.a = [] #Acceleration
        #from cost function
        self.scar = []
        #Inputs
        self.C = []
        self.J = []
        #Track
        self.s = [] #Position along the S-coordinate in frenet frame
        self.x = [] #Global
        self.y = [] #Global
        self.yaw = [] #Yaw angle of the Track
        self.c = []   #Curvature of the Track
        self.ds = []

        # Second model states
        self.S = []  # Position along centerline
        self.D = []  # Deviation from centerline
        self.S_d = []  # Speed along centerline
        self.D_d = []  # speed perp. from centerline

        # inputs
        self.S_dd = []  # Acceleration along centerlin
        self.D_dd = []  # Acceleration perp. centerline

class Track_info:

    def __init__(self):
        # Track
        self.s = []  # Position along the S-coordinate in frenet frame
        self.x = []  # Global
        self.y = []  # Global
        self.yaw = []  # Yaw angle of the Track
        self.c = []  # Curvature of the Track
        self.ds = []



def MPC_calc(csp, mpc_est, ITERATION, track_info):
    mpcp = MPC_path()


    # track info
    s_track = track_info.s
    c = track_info.c
    yaw = track_info.yaw


    X_0simple = [mpc_est.S[1], mpc_est.D[1], mpc_est.S_d[1], mpc_est.D_d[1]]



    n = 4  # States x
    m = 2  # Control signals u
    print "X_0:", X_0simple

    x = cvxpy.Variable(n, N)
    u = cvxpy.Variable(m, N-1)
    slack = cvxpy.Variable(n, N)
    slackSdd = cvxpy.Variable(1,1)
    slackDdd = cvxpy.Variable(1,1)

    cost_matrix = np.eye(n,n)
    Q_slack = cost_matrix * 10
    Q_inputs = 10
    cost = 0.0
    constr = []
    c_bar = []


    for t in range(N-1): #Detta är som att köra MPCn fär alla N states, är de korrekt?

        idx = (np.abs(np.asarray(s_track) - mpc_est.S[t])).argmin()
        c_bar.append(c[idx])
        yaw_prime = (yaw[idx+1]-yaw[idx])/0.05 #ds = 0.05 from csp. I teorin samma som c_bar

        Ad = np.matrix([[1, 0, dt, 0],
                       [0, 1, 0, dt],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

        invert = is_invertible(Ad - np.eye(n, n))
        print "Invertable?: ", invert

        Bd = np.matrix([[0, 0],
                       [0, 0],
                       [dt, 0],
                       [0, dt]])
        v_bar = math.sqrt(mpc_est.S_d[t]**2 + mpc_est.D_d[t]**2)


        #Pedro + Max cost function
        cost += -(c_bar[-1])/(v_bar)*x[1, t]
        #cost += (1 - d_bar[t]*c_bar[-1])/(2*v_bar[t])*x[1, t]**2
        cost += -((1-mpc_est.D[t]*c_bar[-1]))/(v_bar**2) * x[2, t] *100
        cost += sum_squares(Q_slack*slack[:, t])
        cost += slackSdd**2 * Q_inputs
        cost += slackDdd**2 * Q_inputs


        constr += [x[:, t + 1] == Ad * x[:, t] + Bd * u[:, t]]

        constr += [x[1, t+1] <= MAX_ROAD_WIDTH + slack[1, t]]  #Lateral Deviation D
        constr += [x[1, t+1] >= -MAX_ROAD_WIDTH - slack[1, t]]
        constr += [x[2, t+1] <= MAX_SPEED + slack[2, t]] #Velocity S_d
        constr += [x[2, t+1] >= 0 - slack[2, t]]
        constr += [x[3, t+1] <= MAX_SPEED + slack[2, t]] #Velocity D_d
        constr += [x[3, t+1] >= -MAX_SPEED - slack[2, t]]

        constr += [u[0, t] <= MAX_ACCEL + slackSdd]  #Acceleration in S_dd
        constr += [u[0, t] >= -MAX_ACCEL - slackSdd]
        constr += [u[1, t] <= MAX_ACCEL + slackDdd]  #Acceleration in D_dd
        constr += [u[1, t] >= -MAX_ACCEL - slackDdd]

        #slack variables
        constr += [slack[:,t] >= 0]
        constr += [slackSdd >= 0]
        constr += [slackDdd >= 0]

        #print("Constr is DCP:", (x[:, t + 1] == (A * x[:, t] + B * u[:, t])).is_dcp())
        '''
        print("Constr is DCP road with   :", (x[0, t] <= MAX_ROAD_WIDTH).is_dcp(), "and: ", (x[0, t] >= -MAX_ROAD_WIDTH).is_dcp())
        print("Constr is DCP angle error :", (x[1, t] <= MAX_PSI).is_dcp(), "and: ", (x[1, t] >= -math.pi/2).is_dcp())
        print("Constr is DCP velocity    :", (x[2, t] <= MAX_SPEED).is_dcp(), "and: ", (x[2, t] >= 0).is_dcp())
        print("Constr is DCP Kurvature   :", (x[3, t] <= MAX_CURVATURE).is_dcp(), "and: ", (x[3, t] >= -MAX_CURVATURE).is_dcp())
        print("Constr is DCP Acceleration:", (x[4, t] <= MAX_ACCEL).is_dcp(), "and: ", (x[4, t] >= -MAX_ACCEL).is_dcp())
        '''


    constr += [x[:, 0] == X_0simple]
    #constr += [u[:, 0] == U_0]
    print("Constr is DCP:", (x[:, 0] == X_0simple).is_dcp())

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constr)
    print("prob is DCP:", prob.is_dcp())
    print("Cost is DCP:", cost.is_dcp())

    print("curvature of x:", x.curvature)
    print("curvature of u:", u.curvature)
    print("curvature of cost:", cost.curvature)

    # Solve with ECOS.
    #prob.solve(solver=cvxpy.ECOS)
    #print("optimal value with ECOS:", prob.value)

    # Solve with ECOS_BB.
    #prob.solve(solver=cvxpy.ECOS_BB)
    #print("optimal value with ECOS_BB:", prob.value)

    # Solve with SCS.
    prob.solve(solver=cvxpy.SCS)
    print("optimal value with SCS:", prob.value)

    print "Status:", prob.status


    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        #The outputs from the MPC is the real states
        mpcp.psi, mpcp.d, mpcp.v, mpcp.K, mpcp.a, mpcp.s = [], [], [], [], [], []

        mpcp.S = x.value[0, :].tolist()
        mpcp.S = mpcp.S[0]
        mpcp.D = x.value[1, :].tolist() #De riktiga statsen
        mpcp.D = mpcp.D[0]
        mpcp.S_d = x.value[2, :].tolist()
        mpcp.S_d = mpcp.S_d[0]
        mpcp.D_d = x.value[3, :].tolist()
        mpcp.D_d = mpcp.D_d[0]

        mpcp.S_dd = u.value[0, :].tolist()
        mpcp.S_dd = mpcp.S_dd[0]
        mpcp.D_dd = u.value[1, :].tolist()
        mpcp.D_dd = mpcp.D_dd[0]

        diff = mpcp.S[-1] - mpcp.S[-2]
        mpcp.S.append(mpcp.S[-1] + diff)
        mpcp.D.append(mpcp.D[-1])
        mpcp.S_d.append(mpcp.S_d[-1])
        mpcp.D_d.append(mpcp.D_d[-1])

        mpcp.S_dd.append(mpcp.S_dd[-1])
        mpcp.D_dd.append(mpcp.D_dd[-1])

        #Mycket verkar ok till hit!
        print "mpc solution S:", mpcp.S
        print "mpc solution D:", mpcp.D
        print "mpc solution S_d:", mpcp.S_d
        print "mpc solution D_d:", mpcp.D_d
        print "mpc solution S_dd:", mpcp.S_dd
        print "mpc solution D_dd:", mpcp.D_dd
        sS = slack.value[0,:]
        sD = slack.value[1,:]
        sS_d = slack.value[2,:]
        sD_d = slack.value[3,:]

        slackSdd = slackSdd.value
        slackDdd = slackDdd.value
        print "The Slack S variable is:", sS
        print "The Slack D variable is:", sD
        print "The Slack S_d variable is:", sS_d
        print "The Slack D_d variable is:", sD_d

        print "The Slack S_dd variable is:", slackSdd
        print "The Slack D_dd variable is:", slackDdd


    return mpcp

def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

'''
def calc_global_paths(mpc_path, csp): #Beräknar trajectories i globala koordinatsystemet från S och D i Frenet
    #Gör så att denna går från MPC output till plot bart

    for fp in mpc_path:

        # calc global positions
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            iyaw = csp.calc_yaw(fp.s[i])#beräknar yaw Angle i varje punkt som derivatan av kurvan
            di = fp.d[i]
            fx = ix + di * math.cos(iyaw + math.pi / 2.0) #punkten x + närliggande katet 
            fy = iy + di * math.sin(iyaw + math.pi / 2.0)# punkten y + frånliggande katet
            fp.x.append(fx) #lägger alla globala x coordinater i en lista
            fp.y.append(fy) #lägger alla globala y coordinater i en lista

        # calc yaw and ds
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx)) #Yaw angle -> derivatan på kurvan
            fp.ds.append(math.sqrt(dx**2 + dy**2)) # längden på tangent vectorn

        fp.yaw.append(fp.yaw[-1]) #lägger till sista yaw'n en gång till, antagligen för att få samma längd på vektorer
        fp.ds.append(fp.ds[-1])

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i]) #beräknar förändringen i yaw per steg längs sträckan.

    return fplist
'''
'''
def global_vehicle_simulator(mpc_path):

    states = Vehcile_state()
    print "type of mpc_path:", type(mpc_path)
    #Tar bara nästa steg i MPCn som de sanna nästa statet
    #Updatera .s_car!!
    # from cost function
    states.scar = mpc_path.scar
    states.s = np.squeeze(np.asarray((mpc_path.s)))
    # States
    states.psi = np.squeeze(np.asarray(mpc_path.psi))  # Yaw angle of car
    states.psi = states.psi[0]
    states.d = np.squeeze(np.asarray(mpc_path.d))  # Deviation from centerline
    states.d = states.d[0]
    states.v = np.squeeze(np.asarray(mpc_path.v))  # Speed
    states.v = states.v[0]
    # Inputs
    states.K = np.squeeze(np.asarray(mpc_path.K))  # Curvature of car car (trajectory-curvature)
    states.K = states.K[0]
    states.a = np.squeeze(np.asarray(mpc_path.a))  # Acceleration
    states.a = states.a[0]

    vehicle_state = states

    return vehicle_state
'''

def frenet_optimal_planning(csp, mpc_est, ITERATION, track_info):

    mpc_path = MPC_calc(csp, mpc_est, ITERATION, track_info)

    #mpc_path_global = calc_global_paths(mpc_path, csp)#De beräknade trajektorerna görs om från Frenet till globala

    #vehicle_state = global_vehicle_simulator(mpc_path)

    return mpc_path


def generate_target_course(x, y):#tar manuelt inmatate coordinater och skapar ett polynom som blir referens!
    csp = cubic_spline_planner.Spline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)
    d = np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, 0.05) #Skapa 0.1 tät vinkelrät linje
    s_len = s.size
    d_len = d.size
    LUTs = np.zeros((s_len, d_len))
    LUTd = np.zeros((s_len, d_len))
    LUTx = np.zeros((s_len, d_len))
    LUTy = np.zeros((s_len, d_len))

    rs, rx, ry, ryaw, rk = [], [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s) #i_s  = incremental s ->här kan vi göra visualitionsconstraintsen
        rx.append(ix)#Ref x
        ry.append(iy)#Ref y
        ryaw.append(csp.calc_yaw(i_s))#Ref yaw
        rk.append(csp.calc_curvature(i_s))#Ref curvature
        rs.append(i_s)

    LTs, LTd, LTx, LTy, refx, refy, refyaw = [], [], [], [], [], [], []
    s_count, d_count = -1, -1
    for i_ss in s:
        s_count = s_count + 1
        LTs = i_ss
        refx, refy = csp.calc_position(i_ss)
        refyaw = csp.calc_yaw(i_ss)
        for i_dd in d:
            if i_dd == -MAX_ROAD_WIDTH:
                d_count = -1
            d_count = d_count + 1
            LTd = -i_dd
            LTx = refx + i_dd*math.sin(refyaw)
            LTy = refy - i_dd*math.cos(refyaw)
            LUTs[s_count, d_count] = LTs
            LUTd[s_count, d_count] = LTd
            LUTx[s_count, d_count] = LTx
            LUTy[s_count, d_count] = LTy

    plt.plot(LUTx[:,:], LUTy[:,:], LUTx[200,2], LUTy[200,2], 'D')
    plt.plot(LUTs[:,:], -3*MAX_ROAD_WIDTH + LUTd[:,:], LUTs[200, 2], -3*MAX_ROAD_WIDTH + LUTd[200, 2], 'D')
    plt.grid(True)
    plt.show()


    return rs, rx, ry, ryaw, rk, csp, LUTs, LUTd, LUTx, LUTy




def main():
    print(__file__ + " start!!")


    # way points

    #wx = [0.0, 10, 20.0,  30,  40,  80,  85,  90, 95.0, 100, 110, 115, 110, 100, 95, 90, 85, 80, 40, 30, 20, 10,  0, -10]
    #wy = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 5.0,  15,  25,  30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]

    wx = [-30, 0.0, 2.0, 5.0, 7.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 250.0] #, 500.0, 800.0, 1000.0]
    wy = [0.0, 0.0, 0.0, 0.0, 0.0,  0.0,  5.0,  30.0,  30.0,  5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #, 0.0, 0.0, 0.0]
    
    #wx = [-20.0, -40.0, -70.0, -100.0, -120.0,  -140.0,  -150.0,   -160.0, -180.0, -200.0, -180.0, -160.0, -150.0, -140.0, -130.0, -120.0, -90.0, -60.0, -40.0, 0.0, 5.0, 0.0, -15.0, -20.0]
    #wy = [0.0, 0.0,  5.0,  0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 20.0, 40.0, 40.0, 40.0, 45.0, 40.0, 35.0, 40.0, 40.0, 40.0, 40.0, 20.0, 0.0, 0.0, 0.0]

    #wx = [0.0, 1000.0]
    #wy = [0.0, 0.0]

    ts, tx, ty, tyaw, tc, csp, LUTs, LUTd, LUTx, LUTy = generate_target_course(wx, wy)


    # initial state, giving the car a initial position
    track_info = Track_info()
    mpc_s = MPC_path()
    '''
    mpc_s.scar = ts[0]
    # States
    mpc_s.d = [0.0]      # Deviation from centerline
    mpc_s.d += N * mpc_s.d  # Tot [0-N,N]
    mpc_s.e_psi = [0.0]  # Yaw angle deviation of car compared to track
    mpc_s.e_psi += N * mpc_s.e_psi  # Tot [0-N,N]
    mpc_s.v = []
    mpc_s.K = [0.0]   # Curvature of car car(global) (trajectory-curvature, estimated to go straight ahead)
    mpc_s.a = [0.0]  # Acceleration
    mpc_s.K += N * mpc_s.K # Tot [0-(N-1),(N-1)]
    mpc_s.a += N * mpc_s.a # Tot [0-(N-1),(N-1)]

    mpc_s.s = []
    mpc_s.s.append(0)
    for i in range(N+1):
        mpc_s.v.append(30.0+(i*1)*0)   # Speed  # Tot [0-N,N]
    for i in range(N):
        mpc_s.s.append(mpc_s.s[-1] + DT)  # Tot [0-N,N]
    # Inputs
    mpc_s.C = [0.0]  # Sharpness (trajectory-curvature change per change in , estimated to make no change)
    mpc_s.C += (N-1) * mpc_s.C # Tot [0-(N-1),(N-1)]
    mpc_s.J = [0.0]  # Jerk
    mpc_s.J += (N-1) * mpc_s.J# Tot [0-(N-1),(N-1)]
    '''
    # Second model states
    mpc_s.S = [0.0]  # Position along centerline
    mpc_s.S += N * mpc_s.S  # Tot [0-N,N]
    mpc_s.D = [0.0]  # Deviation from centerline
    mpc_s.D += N * mpc_s.D  # Tot [0-N,N]
    mpc_s.S_d = [10.0]  # Speed along centerline
    mpc_s.S_d += N * mpc_s.S_d  # Tot [0-N,N]
    mpc_s.D_d = [0.0]  # speed perp. from centerline
    mpc_s.D_d += N * mpc_s.D_d  # Tot [0-N,N]

    # inputs
    mpc_s.S_dd = [10.0]  # Acceleration along centerlin
    mpc_s.S_dd += (N-1) * mpc_s.S_dd  # Tot [0-(N-1),(N-1)]
    mpc_s.D_dd = [0.0]  # Acceleration perp. centerline
    mpc_s.D_dd += (N-1) * mpc_s.D_dd  # Tot [0-(N-1),(N-1)]



    mpc_est = mpc_s
    initial = mpc_s

    #Track
    track_info.s, track_info.c, track_info.yaw, track_info.x, track_info.y = ts, tc, tyaw, tx, ty

    displacement, travel, x_path, y_path, steering, v, acceleration = [], [], [], [], [], [], []

    area = 100
    for i in range(500):#Antalet gånger koden körs. Bryts när målet är nått!! Detta blir Recursive delen i MPC'n
        ITERATION = i
        print "ITERATION: ", ITERATION

        #saving all values:
        displacement.append(mpc_est.D[1])
        travel.append(mpc_est.S[1])
        acceleration.append(mpc_est.S_dd[1])
        x, y = [], []
        for i in range(N):
            idx, x_coord, y_coord, ix, iy = [], [], [], [], []
            X = np.sqrt(np.square(LUTs - mpc_est.S[i]) + np.square(LUTd - mpc_est.D[i]))
            idx = np.where(X == X.min())
            ix = np.asscalar(idx[0][0])
            iy = np.asscalar(idx[1][0])
            x_coord = LUTx[ix, iy]
            y_coord = LUTy[ix, iy]
            x.append(x_coord)
            y.append(y_coord)

        x_path.append(x[0])
        y_path.append(y[0])


        mpc_path = frenet_optimal_planning(csp, mpc_est, ITERATION, track_info) #The magic



        '''
        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0:
            print("Goal")
            break
        '''
        if show_animation:
            plt.cla()
            '''
            plt.figure(1)
            plt.subplot(231)
            plt.plot(mpc_est.s[:], mpc_est.a, '-ok')
            plt.title('Acceleration predictions')
            plt.ylim(-MAX_ACCEL - 5, MAX_ACCEL + 5)

            plt.subplot(232)
            plt.plot(mpc_path.s, mpc_path.d, '-or')
            plt.title('Displacement predictions')
            plt.ylim(-3, 3)

            plt.subplot(233)
            plt.plot(mpc_est.s, mpc_est.v, '-og')
            plt.title('Velecity predictions')

            plt.subplot(234)
            plt.plot(travel, acceleration, 'k')
            plt.title('Acceleration')
            plt.ylim(-MAX_ACCEL - 5, MAX_ACCEL + 5)

            plt.subplot(235)
            plt.plot(travel, displacement, 'r', LUTs[:,0], LUTd[:,0], 'b', LUTs[:,-1], LUTd[:,-1], 'y')
            plt.title('Displacement')


            plt.subplot(236)
            plt.plot(travel, v, 'g')
            plt.title('Velocity')
            

            plt.savefig('/home/maxahlberg/Pictures/my_new_fig1.png')
            plt.figure(1)
            plt.plot(x_path, y_path, LUTx[:, 0], LUTy[:, 0], 'b', LUTx[:, -1], LUTy[:, -1], 'y', x, y, '-or')
            plt.title("Iteration: " + '{:.2f}'.format(ITERATION) +
                      " v: " + '{:.2f}'.format(mpc_est.v[0])
                      + " K[-1]: " + '{:.2f}'.format(mpc_est.K[-1]))
            plt.xlim(-10, 120)
            plt.ylim(-10, 50)
            plt.savefig('/home/maxahlberg/Pictures/my_new_fig2.png')


            plt.figure(2)
            plt.subplot(231)
            plt.plot(mpc_path.s, mpc_path.d, '-or', mpc_path.s, mpc_est.d, '.g')
            plt.title('Displacement predictions')

            plt.subplot(232)
            plt.plot(mpc_path.s, mpc_path.v, '-ok', mpc_path.s, mpc_est.v, '.g')
            plt.title('Velocity predictions')

            plt.subplot(233)
            #plt.plot(mpc_est.s, mpc_est.v, '-og')
            #plt.title('Velecity predictions')
            plt.plot(mpc_path.s, mpc_path.a, '-ok', mpc_path.s, mpc_est.a, '.g')
            plt.title('Acceleration predictions')

            plt.subplot(234)
            plt.plot(mpc_path.s, mpc_path.K, '-oy', mpc_path.s, mpc_est.K, '.g')
            plt.title('Curvature predictions')

            plt.subplot(235)
            plt.plot(mpc_path.s[:-1], mpc_path.C, '-oy', mpc_path.s[:-1], mpc_est.C, '.g')
            plt.title('Sharpness predictions')

            plt.subplot(236)
            plt.plot(mpc_path.s[:-1], mpc_path.J, '-ok', mpc_path.s[:-1], mpc_est.J, '.g')
            plt.title('Jerk')
            plt.show()
            '''

            plt.figure(1)
            plt.subplot(231)
            plt.plot(mpc_path.S, mpc_path.D, '-or', mpc_path.S, mpc_est.D, '.g')
            plt.title('Displacement predictions')
            plt.ylim(-4, 4)

            plt.subplot(232)
            plt.plot(mpc_path.S, mpc_path.S_d, '-ok', mpc_path.S, mpc_est.S_d, '.g')
            plt.title('S_d')

            plt.subplot(233)
            #plt.plot(mpc_est.s, mpc_est.v, '-og')
            #plt.title('Velecity predictions')
            plt.plot(mpc_path.S, mpc_path.D_d, '-ok', mpc_path.S, mpc_est.D_d, '.g')
            plt.title('D_d')

            plt.subplot(234)


            plt.subplot(235)
            plt.plot(mpc_path.S[:-1], mpc_path.S_dd, '-oy', mpc_path.S[:-1], mpc_est.S_dd, '.g')
            plt.title('S_dd')

            plt.subplot(236)
            plt.plot(mpc_path.S[:-1], mpc_path.D_dd, '-ok', mpc_path.S[:-1], mpc_est.D_dd, '.g')
            plt.title('D_dd')

            plt.figure(2)
            plt.plot(x_path, y_path, LUTx[:, 0], LUTy[:, 0], 'b', LUTx[:, -1], LUTy[:, -1],  x, y, '-or')
            plt.title('X-Y position')
            plt.xlim(0,70)
            plt.show()

            # initial = vehicle_state
            mpc_est = mpc_path

            #plt.pause(0.001)

    print("Finish")
    if show_animation:
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()


if __name__ == '__main__':
    main()
