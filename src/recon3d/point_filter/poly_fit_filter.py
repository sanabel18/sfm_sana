import numpy as np
from numpy.polynomial import polynomial as pl
import scipy.optimize as optimize
from .point_filter import PointFilter  
from utils.logger import get_logger, close_all_handlers
EPS = 1e-4
GSSEPS = 1e-4
DEFAULT_LOGGER = get_logger("PolyFitFilter.log",'.')
class PolyFitFilter():
    '''
    Fit a set of 3D-points with numpy polynomial 
    x = F_X(u)
    y = F_Y(u)
    z = F_Z(u)
    where F_*(u) has the form
    F_*(u) = c0 + c1*u + c2*u**2+ c3*u**3+ ..., depends on polynomial degree chosen
    where u is vector of independent variables
    
    each u[i] can be fine-tuned within bonads [u[i+1], u[i-1]]
    by Golden Section Search Method
    with find-tuned set of u, one will repead PolyFit

    main loop will look likd:

    u = u_inital
    for i < imax:
        fitted_coeff  = PolyFit(x,y,z,u) 
        u_fine_tuned = GSS(u)
        u = u_fine_tuned

    '''
    def __init__(self, poly_degree: float, max_iter:int, init_mode = 'centripetal'):
        self.polydeg = poly_degree       # Degree of polynormial of parametric curve
        self.max_iter = max_iter         # Max. number of iterations
        self.eps = EPS
        self.init_mode = init_mode

    def init_param(self, P, mode, logger=DEFAULT_LOGGER):
        if mode == 'uniform':
            u = self._uniform_param(P)
        elif mode == 'chordlength':
            u = self._chordlength_param(P)
        elif mode == 'centripetal':
            u = self._centripetal_param(P)
        else:
            e = 'mode {} is not supported. Please choose from \'uniform\', \'chordlength\' or \'centripetal\' '
            logger.error(e)
            raise ValueError(e)
        return u
    def _uniform_param(self, P):
        '''
        sample u uniformly
        Args: array(N,3)
        Return: array(N)
        '''
        u = np.linspace(0, 1, len(P))
        return u

    def _chordlength_param(self, P):
        u = self._generate_param(P, alpha=1.0)
        return u

    def _centripetal_param(self, P):
        u = self._generate_param(P, alpha=0.5)
        return u

    def _generate_param(self, P, alpha):
        '''
        sample u accoring to data P with step size: (P[i] - P[i-1])**alpha
        this will create non-uniform sample points
        Args: array(N,3)
        Return: array(N)
        '''
 
        n = len(P)
        u = np.zeros(n)
        u_sum = 0
        for i in range(1,n):
            u_sum += np.linalg.norm(P[i,:]-P[i-1,:])**alpha
            u[i] = u_sum

        return u/max(u)

    def find_min_gss(self, f, a, b, eps = GSSEPS):
        '''
        Find Minimum by Golden Section Search Method
        Return x minimizing function f(x) on interval a,b
        Ref:[https://en.wikipedia.org/wiki/Golden-section_search]

        Args: 
        f: function to be minimized
        a: lower bound of interval
        b: upper bound of interval
        return:
        value between a and b that minimize f(x)
        '''

        # Golden section: 1/phi = 2/(1+sqrt(5))
        R = 0.61803399

        # Num of needed iterations to get precision eps: log(eps/|b-a|)/log(R)
        INV_LOG_R = 1 / np.log(R)
        n_iter = int(np.ceil(INV_LOG_R * np.log(eps/abs(b-a))))
        c = b - (b-a)*R
        d = a + (b-a)*R

        for i in range(n_iter):
            if f(c) < f(d):
                b = d
            else:
                a = c
            c = b - (b-a)*R
            d = a + (b-a)*R

        return (b+a)/2

    def iterative_param(self, P, u, fxcoeff, fycoeff, fzcoeff):

        u_new = u.copy()
        f_u = np.zeros(3)

        #--- Calculate approx. error s(u) related to point P_i
        def calc_s(u):
            f_u[0] = pl.polyval(u, fxcoeff)
            f_u[1] = pl.polyval(u, fycoeff)
            f_u[2] = pl.polyval(u, fzcoeff)

            s_u = np.linalg.norm(P[i]-f_u)
            return s_u

        #--- Find new values u that locally minimising the approximation error (excl. fixed end-points)
        for i in range(1, len(u)-1):

            #--- Find new u_i minimising s(u_i) by Golden search method
            u_new[i] = self.find_min_gss(calc_s, u[i-1], u[i+1])

        return u_new

    def filter(self, pos_xyz, num_sigma = 3):
        '''
        mark data as valid if data pts with error 
        lie within num_sigma*stanard deviations from the mean error.
        i.e. if num_sigma = 3, 
        data with error < np.mean(errors) + 3*np.std(errors)
        will be a valid data

        Args:
        pos_xyz: array(N,3)
        num_sigma: int
        Return: array of valid ids after poly_fit
        '''
        P = np.array(pos_xyz)
        num_data = P.shape[0]
        w = np.ones(num_data)            # Set weights for knot points
        w[0] = w[-1] = 1e6

        #-------------------------------------------------------------------------------
        # Init variables
        #-------------------------------------------------------------------------------
        f_u = np.zeros([num_data,3])
        uu = np.linspace(0,1,num_data)
        f_uu = np.zeros([len(uu),3])
        S_hist = []

        #-------------------------------------------------------------------------------
        # Compute the iterative approximation
        #-------------------------------------------------------------------------------
        for iter_i in range(self.max_iter):

            #--- Initial or iterative parametrization
            if iter_i == 0:
                u = self.init_param(P, self.init_mode)
            else:
                u = self.iterative_param(P, u, fxcoeff, fycoeff, fzcoeff)

            #--- Compute polynomial approximations and get their coefficients
            fxcoeff = pl.polyfit(u, P[:,0], self.polydeg, w=w)
            fycoeff = pl.polyfit(u, P[:,1], self.polydeg, w=w)
            fzcoeff = pl.polyfit(u, P[:,2], self.polydeg, w=w)
            #--- Calculate function values f(u)=(fx(u),fy(u),fz(u))
            f_u[:,0] = pl.polyval(u, fxcoeff)
            f_u[:,1] = pl.polyval(u, fycoeff)
            f_u[:,2] = pl.polyval(u, fzcoeff)

            #--- Calculate fine values for ploting
            f_uu[:,0] = pl.polyval(uu, fxcoeff)
            f_uu[:,1] = pl.polyval(uu, fycoeff)
            f_uu[:,2] = pl.polyval(uu, fzcoeff)


            #--- Errors of init parametrization
            error = P - f_u
            error_norm = np.linalg.norm(error, axis=1)
            #print("errors: {}".format(error_norm))   

            #--- Total error of approximation S for iteration i
            S = 0
            for j in range(len(u)):
                S += w[j] * np.linalg.norm(P[j] - f_u[j])

            S_hist.append(S)

            #--- Stop iterating if change in error is lower than desired condition
            if iter_i > 0:
                S_change = S_hist[iter_i-1] / S_hist[iter_i] - 1
                #print('iteration:%3i, approx.error: %.4f (%f)' % (iter_i, S_hist[iter_i], S_change))
                if S_change < self.eps:
                    break
        
        #===============================================
        #Mark big residual errors 
        thr = np.mean(error_norm) + num_sigma*np.std(error_norm)
        valid_index = np.where(error_norm < thr)[0]
        #===============================================
  
        return valid_index


